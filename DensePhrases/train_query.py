import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string
import faiss
from torch import amp
import wandb

from time import time
from tqdm import tqdm
from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, \
                                          drqa_metric_max_over_ground_truths, normalize_answer
from eval_phrase_retrieval import evaluate
from densephrases import Options

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train_query_encoder(args, mips=None):
    # assert sent option is correct
    assert (args.label_strat == "sent") == args.return_sent, "return_sent argument must match with label_strat 'sent'"

    # Freeze one for MIPS
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    target_encoder, tokenizer, _ = load_encoder(device, args)


    # MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Optimizer setting
    def is_train_param(name):
        if name.startswith("phrase_encoder"):
            logger.info(f'freezing {name}')
            return False
        if name.endswith(".embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.01,
        }, {
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    step_per_epoch = math.ceil(len(load_qa_pairs(args.train_path, args)[1]) / args.per_gpu_train_batch_size)
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"Train for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
     )
    eval_steps = math.ceil(len(load_qa_pairs(args.dev_path, args)[1]) / args.eval_batch_size)
    logger.info(f"Test takes {eval_steps} iterations")

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    global_step = 0
    metric = {
        "train_loss":1e9,
        "train_acc@1":-1000.0,
        "train_acc@k":-1000.0,
        "dev_acc@1":-1000.0,
        "dev_acc@k":-1000.0,
        "dev_recall@k":-1000.0
    }

    # Training
    total_loss = 0.0
    total_accs = []
    total_accs_k = []

    for ep_idx in range(int(args.num_train_epochs)):
        # Load training dataset
        q_ids, questions, answers, titles = load_qa_pairs(args.train_path, args, shuffle=True)
        pbar = tqdm(get_top_phrases(
            mips, q_ids, questions, answers, titles, target_encoder, tokenizer,
            args.per_gpu_train_batch_size, args)
        )

        for step_idx, (q_ids, questions, answers, titles, outs) in enumerate(pbar):
            train_dataloader, _, _ = get_question_dataloader(
                questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
            )
            svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args)

            target_encoder.train()
            svs_t = torch.Tensor(svs).to(device)
            evs_t = torch.Tensor(evs).to(device)
            tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
            p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts]

            # Train query encoder
            assert len(train_dataloader) == 1
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                loss, accs = target_encoder.train_query(
                    input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                    start_vecs=svs_t,
                    end_vecs=evs_t,
                    targets=tgts_t,
                    p_targets=p_tgts_t,
                )

                # Optimize, get acc and report
                if loss is not None:
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    total_loss += loss.mean().item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    target_encoder.zero_grad()

                    pbar.set_description(
                        f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f}, acc: {sum(accs)/len(accs):.3f}"
                    )
                    wandb.log( 
                            {"train_loss": loss.mean().item(), "train_acc@1(step)": sum(accs)/len(accs), "learning_rate":optimizer.param_groups[0]['lr']} , step=global_step,)


                if accs is not None:
                    total_accs += accs
                    total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                else:
                    total_accs += [0.0]*len(tgts_t)
                    total_accs_k += [0.0]*len(tgts_t)

                global_step += 1
                # Save best model
                if args.save_steps and not global_step % args.save_steps:
                    metric["train_loss"] = total_loss/args.save_steps
                    metric["train_acc@1"] = sum(total_accs)/len(total_accs)
                    metric["train_acc@k"] = sum(total_accs_k)/len(total_accs_k)
                    save_model(args, global_step, metric, target_encoder)

                    logger.info(
                        f"Avg train loss ({global_step} iterations): {total_loss/args.save_steps:.2f} | train " +
                        f"train_acc@1: {sum(total_accs)/len(total_accs):.3f} | train_acc@{args.top_k}: {sum(total_accs_k)/len(total_accs_k):.3f}"
                    )
                    wandb.log( 
                          {"train_acc@1(avg)": sum(total_accs)/len(total_accs), f"train_acc@{args.top_k}": sum(total_accs_k)/len(total_accs_k), "train_loss_avg":total_loss/args.save_steps} , step=global_step,)
                    total_loss = 0.0
                    total_accs = []
                    total_accs_k = []
                
                if args.eval_steps and not global_step % args.eval_steps:
                    # Evaluation
                    dev_top_1_acc, dev_top_k_acc, dev_top_k_recall = dev_eval(args, mips, target_encoder, tokenizer)
                    metric["dev_acc@1"] = dev_top_1_acc
                    metric["dev_acc@k"] = dev_top_k_acc
                    metric["dev_recall@k"] = dev_top_k_recall
                    logger.info(f"Develoment set dev_acc@1: {dev_top_1_acc:.3f}, dev_acc@{args.dev_top_k}: {dev_top_k_acc:.3f}, dev_recall@{args.dev_top_k}: {dev_top_k_recall:.3f}")
                    wandb.log( 
                            {"dev_acc@1": dev_top_1_acc, f"dev_acc@{args.dev_top_k}": dev_top_k_acc, f"dev_recall@{args.dev_top_k}": dev_top_k_recall} , step=global_step,)
    last_steps = global_step % args.save_steps
    if last_steps:
        metric["train_loss"] = total_loss/last_steps
        metric["train_acc@1"] = sum(total_accs)/len(total_accs)
        metric["train_acc@k"] = sum(total_accs_k)/len(total_accs_k)
        save_model(args, global_step, metric, target_encoder)
        logger.info(
            f"Avg train loss ({global_step} iterations): {total_loss/last_steps:.2f} | train " +
            f"train_acc@1: {sum(total_accs)/len(total_accs):.3f} | train_acc@{args.top_k}: {sum(total_accs_k)/len(total_accs_k):.3f}"
        )
        wandb.log( 
                {"train_acc@1": sum(total_accs)/len(total_accs), f"train_acc@{args.top_k}": sum(total_accs_k)/len(total_accs_k), "train_loss_avg":total_loss/last_steps} , step=global_step,)
    logger.info(f"model saved into {args.output_dir}")

def dev_eval(args, mips, target_encoder, tokenizer):
    is_sent = args.return_sent
    args.return_sent = True
    q_ids, questions, answers, titles = load_qa_pairs(args.dev_path, args)
    pbar = tqdm(get_top_phrases(
            mips, q_ids, questions, answers, titles, target_encoder, tokenizer,
            args.eval_batch_size, args, is_eval=True)
        )
    top_k_boolean = []
    for step_idx, (q_ids, questions, answers, titles, outs) in enumerate(pbar):
        top_k_boolean += [
            [any(answer in phrase['context'] for answer in answer_set) for phrase in phrase_group]
            for phrase_group, answer_set in zip(outs, answers)
        ]
    top_k_recall = [sum(i)/(args.dev_top_k*2) for i in top_k_boolean]
    top_k_acc = [any(i) for i in top_k_boolean]
    top_1_acc = [i[0] for i in top_k_boolean]
    args.return_sent = is_sent
    return sum(top_1_acc)/len(top_1_acc), sum(top_k_acc)/len(top_k_acc), sum(top_k_recall)/len(top_k_recall)

def save_model(args, global_step, metric, model):
    save_path = os.path.join(args.output_dir, f"step_{global_step}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    metric_save_path = os.path.join(save_path, "metric.json")
    with open(metric_save_path, "w") as f:
        json.dump(metric, f)
    logger.info(f"Saved best model at step {global_step} into {save_path}")


    # modifyable
def get_top_phrases(mips, q_ids, questions, answers, titles, query_encoder, tokenizer, batch_size, args, is_eval=False):
    # Search
    step = batch_size
    phrase_idxs = []
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = query2vec(questions[q_idx:q_idx+step]) 
        # batch size query vector list 
        # out[0]: start vector 
        # out[1]: end vector 
        # out[2]: decoded sentence
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1) #shape: (batch_size, 2*d_model)

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.dev_top_k if is_eval else args.top_k,
            return_idxs=True, max_answer_length=args.max_answer_length,
            aggregate=args.aggregate, agg_strat=args.agg_strat,
            return_sent = True if args.label_strat == "sent" else False
        )
        yield (
            q_ids[q_idx:q_idx+step], questions[q_idx:q_idx+step], answers[q_idx:q_idx+step],
            titles[q_idx:q_idx+step], outs
        )

    # modifyable
def annotate_phrase_vecs(mips, q_ids, questions, answers, titles, phrase_groups, args):
    assert mips is not None
    batch_size = len(answers)

    # Phrase groups are in size of [batch, top_k, values]
    # phrase_groups = [[(
    #     out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer'],
    #     out_['start_vec'], out_['end_vec'], out_['context'], out_['title'])
    #     for out_ in out] for out in outs
    # ]
    dummy_group = {
        'doc_idx': -1,
        'start_idx': 0, 'end_idx': 0,
        'answer': '',
        'start_vec': np.zeros(768),
        'end_vec': np.zeros(768),
        'context': '', 'title': ['']
    }

    # Pad phrase groups (two separate top-k coming from start/end, so pad with top_k*2)
    for b_idx, phrase_idx in enumerate(phrase_groups):
        while len(phrase_groups[b_idx]) < args.top_k*2:
            phrase_groups[b_idx].append(dummy_group)
        assert len(phrase_groups[b_idx]) == args.top_k*2

    # Flatten phrase groups
    flat_phrase_groups = [phrase for phrase_group in phrase_groups for phrase in phrase_group]
    doc_idxs = [int(phrase_group['doc_idx']) for phrase_group in flat_phrase_groups]
    start_vecs = [phrase_group['start_vec'] for phrase_group in flat_phrase_groups]
    end_vecs = [phrase_group['end_vec'] for phrase_group in flat_phrase_groups]
    
    # stack vectors
    start_vecs = np.stack(start_vecs)
    end_vecs = np.stack(end_vecs)
    zero_mask = np.array([[1] if doc_idx >= 0 else [0] for doc_idx in doc_idxs])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Reshape
    start_vecs = np.reshape(start_vecs, (batch_size, args.top_k*2, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, args.top_k*2, -1))

    # Dummy targets
    targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]
    p_targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]

    # TODO: implement dynamic label_strategy based on the task name (label_strat = dynamic)

    # Annotate for L_phrase / L_sent
    if 'phrase' in args.label_strat.split(','):
        match_fns = [
            drqa_regex_match_score if args.regex or ('trec' in q_id.lower()) else drqa_exact_match_score for q_id in q_ids
        ]
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, phrase['answer'], answer_set) for phrase in phrase_group]
            for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]
    elif 'sent' in args.label_strat.split(','):
        targets = [
            [any(answer in phrase['context'] for answer in answer_set) for phrase in phrase_group]
            for phrase_group, answer_set in zip(phrase_groups, answers)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Annotate for L_doc
    if 'doc' in args.label_strat.split(','):
        p_targets = [
            [any(phrase['title'][0].lower() == tit.lower() for tit in title) for phrase in phrase_group]
            for phrase_group, title in zip(phrase_groups, titles)
        ]
        p_targets = [[ii if val else None for ii, val in enumerate(target)] for target in p_targets]

    return start_vecs, end_vecs, targets, p_targets


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_qsft_options()
    args = options.parse()
    
    wandb.init(project=args.project, entity=args.entity, name=args.run_name, mode="online" if args.wandb else "disabled")
    wandb.config.update(args)

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'train_query':
        # Train
        mips = load_phrase_index(args)
        train_query_encoder(args, mips)

        # # Eval
        # args.load_dir = args.output_dir
        # logger.info(f"Evaluating {args.load_dir}")
        # args.top_k = 10
        # evaluate(args, mips)

    else:
        raise NotImplementedError
