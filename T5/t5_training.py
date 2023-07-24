from transformers import AutoTokenizer, AutoModel
#TODO 바꾸기
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, Dataset, load_from_disk
import numpy as np
import torch
import multiprocessing
import evaluate
#device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#dataset 준비 #수정해야할 부분(받아서 만들기)
dataset = load_from_disk('t5_dataset')
#수정해야 할 부분
model_path = "t5-large" #model search(정민님은 small)
max_token_length = 64
tokenizer = AutoTokenizer.from_pretrained(model_path)

def convert_examples_to_features(examples):
    # 수정해야함
    model_inputs = tokenizer(examples['original'],
                             text_target=examples['rephrased'], 
                             max_length=max_token_length, truncation=True)
    
    return model_inputs

NUM_CPU = multiprocessing.cpu_count() 
tokenized_datasets = dataset.map(convert_examples_to_features, 
                                 batched=True, 
                                 remove_columns=dataset["train"].column_names,
                                 num_proc=NUM_CPU)
#모델 준비
model = AutoModel.from_pretrained(model_path).to(device)
#collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
#metric 준비(수정해야할 부분)
metric = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    
    return result
#trainer 준비
training_args = Seq2SeqTrainingArguments(
    output_dir="./data/output",
    learning_rate=0.0005,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_strategy="no",
    predict_with_generate=True,
    fp16=False,
    gradient_accumulation_steps=2,
    report_to="none" # Wandb 로그 끄기
)
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("./results")