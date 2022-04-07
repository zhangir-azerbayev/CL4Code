from codex import openai_call
from cl.data.dataset import read_gsm8k
from cl.execution import semisafe_evaluate
from tqdm import tqdm
import re
import random
import json
random.seed(10)
k = 20
temp = 0.2
num_examples = 300

prompt = open("prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")
random.shuffle(train_data)
log = []

for instance in tqdm(train_data[:num_examples]): 
    label = instance.answer
    inputs = [prompt + instance.text for _ in range(k)]
    outputs = openai_call(inputs, temp=temp)

    re_key = '\nanswer.*?\n'

    bodies = [completion[:re.search(re_key, completion).span()[1]]
        if re.search(re_key, completion) else completion
        for completion in outputs]
    
    answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]

    passed_lst = [(abs((answer - label)/label) < 0.01) 
                    if isinstance(answer, float) else False 
                    for answer in answers]
    
    if True in passed_lst: 
        gold_code = bodies[passed_lst.index(True)]
        passed = 1
    else: 
        gold_code = False 
        passed = 0 

    log.append({"task_id": instance.task_id, 
                "text": instance.text, 
                "answer": instance.answer, 
                "gold_solution": gold_code,
                "pass@k": passed})

                
with open("codex_gsm8k_log.json", "w") as fle: 
    json.dump(log, fle)

    





