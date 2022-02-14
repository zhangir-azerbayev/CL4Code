import sys 
import os
import yaml 
import math

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

def update_solved(model, solved, train_set, temp, k, few_shot=False): 
    few_shot_prompt = ""

    if few_shot: 
        few_shot_prompt = "\n\n".join(random.sample(train_set, 4))

    for idx, instance in enumerate(tqdm(train_set)): 
        label = instance.answer
        prompt = few_shot_prompt + instance.text

        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt", device=device)
        prompt_length = torch.numel(encoded_prompt)


        with torch.no_grad():
            out = model.generate(
                    input_ids=encoded_prompt,
                    do_sample=True,
                    temperature=temp,
                    max_new_tokens = max_length,
                    pad_token_id = tokenizer.eos_token_id,
                    num_return_sequences=samples
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
            new_instance = instance 
            instnace.

        per_temp_result = dict()
        for k in ks:
            per_temp_result[f"pass{k}"] = pass_k(passed_lst, k)

        if True in passed_lst:
            best_completion = bodies[passed_lst.index(True)]
        else:
            # This is not the best way
            best_completion = bodies[0]

        per_temp_result["best_completion"] = best_completion

        result[f"temp{temp}"] = per_temp_result

    results.append(result)






config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
param_count = cfg['param_count']
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
max_length = cfg['max_length']
epochs = cfg['epochs']
batch_size = cfg['batch_size']
optim = cfg['optimizer']
lr = optim['lr']
weight_decay = optim['weight_decay']

results_dir = f"train_results/{experiment_name})"
os.mkdir(results_dir)

model_name = f"EleutherAI/gpt-neo-{param_count}"
model = GPTNeoForCausalLM.from_pretrained(model_name, device=device)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

all_data_list = read_mathqapython('../data/mathqapython_train.json')


