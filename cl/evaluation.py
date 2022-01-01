import sys
import re
import random
import yaml
import json
import os
import itertools
from tqdm import tqdm 
random.seed(1)

import numpy as np 

import torch
from torch.utils.data import DataLoader, BatchSampler

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from data.dataset import read_mathqapython, MathQAPython 

from execution import semisafe_evaluate

# pass_k function 
def pass_k(lst, k): 
    """
    lst: Boolean list 
    k: value of pass@k to calculate. 
    """
    n = len(lst)
    c = sum(lst)
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / 
                        np.arange(n-c+1, n+1))

# Extract info from config file 
config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
few_shot = cfg['few_shot']
data_path = cfg['data_path']
model_path = cfg['model_path']
param_count = cfg['param_count']
device = cfg['device']
batch_size = cfg['batch_size']
max_length = cfg['max_length']
temp_ks_samples = cfg['temp_ks_samples']

out_dir = "results_evaluation/" + experiment_name + "/"
os.mkdir(out_dir)

model_name = f"EleutherAI/gpt-neo-{param_count}"

# Load data 
print("loading data...")

eval_set = read_mathqapython(data_path)
if few_shot == 1: 
    train_set = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained(model_name)[:10]
tokenizer.pad_token = tokenizer.eos_token 

# Load model 
print("loading model...")
if few_shot == 1: 
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
else: 
    model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)

results = []
for batch in tqdm(BatchSampler(eval_set, batch_size, drop_last=False)): 
    instances = list(batch)
    texts = [instance.text for instance in instances]

    if few_shot==1: 
        for i in instances: 
            examples = random.sample(train_set, 3)
            few_shot_prompt = ""
            for example in examples: 
                few_shot_prompt = (few_shot_prompt + example.text + "\n" +
                        example.code + "\n\n")
            texts[i] = few_shot_prompt + texts[i]

    encoded_dict = tokenizer.encode(texts, return_tensors="pt").to(device)
    prompt_length = torch.numel(encoded_dict["input_ids"][0])
 
    batch_results = [dict() for _ in range(batch_size)]
    for n in batch results:
        batch_results[n]["task_id"] = instances[n].task_id

    for triple in temp_ks_samples: 
        temp = triple['temp']
        ks = triple['ks']
        samples = triple['samples']
        
        with torch.no_grad(): 
            out = model.generate(
                    **encoded_dict
                    do_sample=True, 
                    temperature=temp, 
                    max_new_tokens = max_length, 
                    pad_token_id = tokenizer.eos_token_id, 
                    num_return_sequences=samples
            )
        
        generated_ids = [ids[prompt_length:] for ids in out]
        untrunced_bodies = [tokenizer.decode(sample, skip_special_tokens=True)
                    for sample in generated_ids]
        
        # Now calculate results by instance
        re_key = '^answer.*?\n'
        for n, i in enumerate(range(0, samples*batch_size, samples)):
            this_instance = untrunced_bodies[i:i+samples]

            bodies = [completion[:re.search(re_key, completion).span()[1]]
                    if re.search(re_key, completion) else completion 
                    for completion in this_instance]

            answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]
            label = instances[n].answer

            passed_lst = [(abs((answer - label)/label) < 0.01)
                    if isinstance(answer, float) else False for answer in answers]

            for k in ks: 
                batch_results[n][f"temp{temp}"][f"pass{k}"] = pass_k(passed_lst, k)

            if True in passed_lst: 
                best_completion = bodies[passed_lst.index(True)]
            else: 
                # This is not the best way
                best_completion = bodies[0]

            batch_results[n][f"temp{temp}"]["best_completion"] = best_completion



    results = results + batch_results

to_dump = dict()

# compute summary statistics

summary = dict()
for triple in temp_ks_samples: 
    per_temp = dict()
    temp = triple["temp"]
    ks = triple["ks"]
    for k in ks: 
        summation = sum([result[f"temp{temp}"][f"pass{k}"] for result in results])
        avg = summation/len(results) 
        per_temp[f"pass{k}"] = avg 

    summary[f"temp{temp}"] = per_temp

to_dump["summary"] = summary 
to_dump["instances"] = results


with open(out_dir + "results.json", "w") as f: 
    json.dump(to_dump, f, indent=4, separators=(',', ': '))

with open(out_dir + "config.yml", "w") as f: 
    yaml.dump(cfg, f)
