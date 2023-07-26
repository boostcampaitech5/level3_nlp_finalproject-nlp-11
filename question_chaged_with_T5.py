import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def changed_with_t5(query):
    set_seed(42)

    model = T5ForConditionalGeneration.from_pretrained('/opt/ml/level3_nlp_finalproject-nlp-11/T5/t5_reparaphrase')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)

    text =  "paraphrase: " + query + " </s>"


    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        top_k=120,
        top_p=0.98
    )
    changed_query = tokenizer.decode(beam_outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(changed_query)
    return changed_query