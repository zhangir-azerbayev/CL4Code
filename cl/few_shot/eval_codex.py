from codex import openai_call
from cl.data.dataset import read_gsm8k
from cl.execution import semisafe_evaluate
from tqdm import tqdm
import re
import random
import json
import openai
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=1, period=60)
def call_api(engine, prompt, max_tokens, n, temperature): 
    return openai.Completion.create(engine=engine, 
            prompt=prompt, max_tokens=max_tokens, n=n, 
            temperature=temperature)



random.seed(20)
k = 20
temp = 0.2
num_examples = 300

prompt = open("prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")
random.shuffle(train_data)
log = []

for instance in tqdm(train_data[:num_examples]): 
    label = instance.answer
    input_seq = prompt + instance.text 


    outputs = call_api(engine="code-davinci-002", 
                       prompt=input_seq, 
                       max_tokens=100, 
                       n=k, 
                       temperature=temp
                       )
    outputs = [output["text"] for output in outputs["choices"]]

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

    pass_1 = sum(passed_lst)/len(passed_lst)

    log.append({"task_id": instance.task_id, 
                "text": instance.text, 
                "answer": instance.answer, 
                "gold_solution": gold_code,
                "passk": passed, 
                "pass1": pass_1, 
                "passed_lst": passed_lst})


num_passed = sum([x["pass@k"] for x in log])
pass_k = num_passed/num_examples

pass_1 = sum([x["pass1"] for x in log])/num_examples


to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open("codex_gsm8k_log_1.json", "w") as fle: 
    json.dump(to_dump, fle)

    





