import sys 
import os
import yaml 
import json
import math
import random
import re

from tqdm import tqdm

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter 

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import get_parameter_names

from cl.data.dataset import read_mathqapython, MathQAPython

from cl.execution import semisafe_evaluate

def change_code(instance, code): 
    instance.set_code(code)
    return instance


def train_model(model, labelled_examples, training_run_name): 
    train_set = MathQAPython(labelled_examples, tokenizer, max_length)


    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]



    training_args = TrainingArguments(output_dir=f"./results_train/{experiment_name}/MLElogs/{training_run_name}",
                                      num_train_epochs=epochs_per_step,
                                      per_device_train_batch_size=train_batch_size, 
                                      logging_steps=500,
                                      save_steps=max_instances,
                                      warmup_steps = 100, 
                                      )

    def data_collator(data):
        return {'input_ids': torch.stack([f[0] for f in data]),
                'attention_mask': torch.stack([f[1] for f in data]), 
                'labels': torch.stack([f[0] for f in data])
               }

    print(f"###############{training_run_name}#################")
    Trainer(model=model, args=training_args, train_dataset=train_set, 
            data_collator=data_collator).train()

    return model 


"""
solved is a set of indices
solutions is an array
"""
def update_solved(model, solved, solutions, train_set, temp, num_samples, few_shot=False): 
    print("#############Updating S_k#################")
    for idx, instance in enumerate(tqdm(train_set)): 
        if idx not in solved: 
            label = instance.answer

            encoded_prompt = tokenizer.encode(instance.text, return_tensors="pt").to(device)
            prompt_length = torch.numel(encoded_prompt)


            with torch.no_grad():
                out = model.generate(
                        input_ids=encoded_prompt,
                        do_sample=True,
                        temperature=temp,
                        max_new_tokens = max_length,
                        pad_token_id = tokenizer.eos_token_id,
                        num_return_sequences=num_samples
                )

            generated_ids = [ids[prompt_length:] for ids in out]
            untrunced_bodies = [tokenizer.decode(sample, skip_special_tokens=True)
                        for sample in generated_ids]

            re_key = '^answer.*?\n'
            bodies = [completion[:re.search(re_key, completion).span()[1]]
                    if re.search(re_key, completion) else completion
                    for completion in untrunced_bodies]

            answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]

            passed_lst = [(abs((answer - label)/label) < 0.01)
                    if isinstance(answer, float) else False for answer in answers]


            if True in passed_lst:
                best_completion = bodies[passed_lst.index(True)]
                solved.add(idx)
                solutions[idx] = best_completion

    return solved, solutions


                

config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
param_count = cfg['param_count']
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
device = int(cfg['devices'])
max_length = cfg['max_length']
epochs_per_step = cfg['epochs']
train_batch_size = cfg['batch_size']
weight_decay = cfg['weight_decay']
num_iters = cfg['num_iters']

results_dir = f"results_train/{experiment_name}"
os.mkdir(results_dir)

model_name = f"EleutherAI/gpt-neo-{param_count}"
model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Just doing 2000 examples 
max_instances = 3000
all_data_list = read_mathqapython('../data/mathqapython_train.json')
random.shuffle(all_data_list)
all_data_list = all_data_list[:max_instances]

# Seed with 100 labelled examples 
num_seeds = 100
solved = set(random.sample(range(max_instances), num_seeds))

solutions = [None for _ in range(max_instances)]
for i in solved: 
    solutions[i] = all_data_list[i].code 


for i in range(num_iters): 
    labelled_examples = [change_code(all_data_list[i], solutions[i]) for i in solved]

    model = train_model(model, labelled_examples, f"MLE{i}")


    solved, solutions = update_solved(model, 
                                     solved, 
                                     solutions, 
                                     all_data_list, 
                                     0.2, 
                                     1
                                     )

    print("NUMBER SOLVED: ", len(solved))

    
    with open(f"results_train/{experiment_name}/S{i}.json", "w") as fle: 
        json.dump({"num solved": len(solved), 
                   "solutions": [{"idx": i, "prompt": all_data_list[i].text, "solution": solutions[i]} 
                                    for i in solved]
                  }, fle, indent=4)
