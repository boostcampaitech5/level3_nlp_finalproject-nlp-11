from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
import torch
import tqdm
from typing import List,Optional
def changed_with_t5(
        #변환 시켜주는 query list
        input_text: List[str],
        #변환시켜줄 T5 model 이름 입력(small, base, large 중 택 1)
        model_type: Optional[str]=None) -> List[str]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = f"jaekwanyda/T5_{model_type}_make_natural" #huggingface 모델 불러오기
    config = AutoConfig.from_pretrained('/opt/ml/level3_nlp_finalproject-nlp-11/T5/config.json') #config에 문제가 있으면 huggingface에서 config 직접 받아 경로 설정해주기!
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,config=config).to(device)
    #TODO: batch_size 조절해보기
    train_data = torch.utils.data.DataLoader(input_text, batch_size=512)
    results=[]
    for _,td in enumerate(tqdm.tqdm(train_data)):
        inputs = tokenizer(td, return_tensors="pt", padding=True).to(device)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
        )
        result=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        results.append(result)
    #TODO: 이 부분 debug 하면서 고쳐보기
    #TODO: cuda에서 model 내리기!
    r = []
    for result in results:
        r+=result
    return r