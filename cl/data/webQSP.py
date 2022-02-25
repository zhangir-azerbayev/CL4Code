import torch
import json
from pathlib import Path

# single data instance. 
# fields included/kept from original dataset subject to change
class WebQSPInstance():
    def __init__(self, 
                 question, 
                 dsl_code, 
                 answer, 
                 task_id):
        # question is a string
        self.question = question
        # list of sparql queries
        self.dsl_code = dsl_code
        # answer is a list of strings
        self.answer = answer 
        self.task_id = task_id

def read_webqsp(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        webqsp_list = json.load(f)

    instance_list = [WebQSPInstance(**dct) for dct in webqsp_list]

    return instance_list

class WebQSP(torch.utils.data.Dataset):
    def __init__(self, instance_list, tokenizer, max_length):
        self.data = instance_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        instance = self.data[idx]
        text = instance.question
        answer = instance.answer

        text_encode = self.tokenizer(text,
                                        max_length=self.max_length, truncation=True,
                                        padding='max_length', return_tensors='pt')
        ids = text_encode['input_ids'].squeeze()
        mask = text_encode['attention_mask'].squeeze()

        return ids.long(), mask.long(), instance.answer

    def __len__(self):
        return len(self.data)
