import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5_model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained('/opt/ml/level3_nlp_finalproject-nlp-11/T5/t5_reparaphrase')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = self.model.to(self.device)
        self.set_seed(42)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def changed_with_t5(self, query):
        text = "paraphrase: " + query + " </s>"
        max_len = 256

        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            top_k=50,
            top_p=0.98
        )
        changed_query = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(changed_query)
        return changed_query

# Example usage:
if __name__ == "__main__":
    t5 = T5_model()
    query = "Your input sentence here."
    changed_query = t5.changed_with_t5(query)