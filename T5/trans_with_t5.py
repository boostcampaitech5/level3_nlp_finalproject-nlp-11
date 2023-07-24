from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "jaekwanyda/T5_small_make_natural" #허깅
device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,config=config).to(device)
def trans_with_T5(input_text, max_token_length = 64):
    train_data = torch.utils.data.DataLoader(input_text, batch_size=1024)
    results=[]
    for i_bacth,td in enumerate(tqdm.tqdm(train_data)):
        inputs = tokenizer(td, return_tensors="pt", padding=True).to(device)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
        )
        result=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        results.append(result)
    return results