import torch 
import json 
from pathlib import Path 

class MathQAInstance(): 
    def __init__(self, 
                 text, 
                 code, 
                 dsl_code, 
                 reasoning, 
                 answer, 
                 task_id): 
        self.text = text
        self.code = code
        self.dsl_code = dsl_code
        self.reasoning = reasoning
        self.answer = answer
        self.task_id = task_id

    def set_code(self, code): 
        self.code = code 


def read_mathqapython(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        mathqapython_list = json.load(f)

    instance_list = [MathQAInstance(**dct) for dct in mathqapython_list]

    return instance_list


class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, instance_list, tokenizer, max_length): 
        self.data = instance_list 
        self.tokenizer = tokenizer 
        self.max_length = max_length
    

    def __getitem__(self, idx): 
        instance = self.data[idx]
        full_text = instance.text + '\n' + instance.code
        answer = instance.answer

        full_text_encode = self.tokenizer(full_text, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        ids = full_text_encode['input_ids'].squeeze()
        mask = full_text_encode['attention_mask'].squeeze()

        return ids.long(), mask.long(), answer


    def __len__(self): 
        return len(self.data) 
