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
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

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


def train_model(model, tokenizer, labelled_examples, training_run_name, cfg):

    train_set = MathQAPython(labelled_examples, tokenizer, cfg["max_length"])


    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    lr_lambda = lambda step: 1
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


    experiment_name = cfg["experiment_name"]
    steps_per_epoch = len(train_set)//cfg["train_batch_size"] + 1

    training_args = TrainingArguments(output_dir=f"./results_train/{experiment_name}/MLElogs/{training_run_name}",
                                      num_train_epochs=cfg["epochs"],
                                      per_device_train_batch_size=cfg["train_batch_size"], 
                                      logging_steps=steps_per_epoch,
                                      save_steps=cfg["epochs"]*steps_per_epoch,
                                      )

    def data_collator(data):
        return {'input_ids': torch.stack([f[0] for f in data]),
                'attention_mask': torch.stack([f[1] for f in data]), 
                'labels': torch.stack([f[0] for f in data])
               }

    print(f"###############{training_run_name}#################")
    Trainer(model=model, args=training_args, train_dataset=train_set, 
            data_collator=data_collator, 
            optimizers=(optimizer, scheduler)).train()

    return model 


"""
solved is a set of indices
solutions is an array
"""
def update_solved(model, 
                  tokenizer,
                  solved, 
                  solutions, 
                  train_set, 
                  temp, 
                  cfg,
                  ): 
    print("#############Updating S_k#################")
    max_text_length = 200
    max_new_tokens = 150
    device = "cuda:" + cfg["devices"]

    inference_dataset = [{"idx": i, 
                          "text": instance.text, 
                          "answer": instance.answer}
                          for i, instance in enumerate(train_set)
                          if i not in solved]

 
    dataloader = DataLoader(inference_dataset, 
            batch_size=cfg["inference_batch_size"], drop_last=False)

    for batch in tqdm(dataloader): 
        batch_length = len(batch["text"])
        encoded_texts = tokenizer(batch["text"], return_tensors="pt",
            max_length=max_text_length, padding='max_length', 
            truncation=True).to(device)
 
        outputs = model.generate(**encoded_texts, 
                                 do_sample=True, 
                                 temperature=temp, 
                                 max_new_tokens = max_new_tokens, 
                                 pad_token_id=tokenizer.eos_token_id, 
                                 num_return_sequences = cfg["inference_num_samples"],
                                )

        outputs = torch.reshape(outputs, 
                (batch_length, cfg["inference_num_samples"], -1))

        for idx, label, outs in zip(batch["idx"], batch["answer"], outputs): 
            generated_ids = [ids[max_text_length:] for ids in outs]
            untrunced_bodies = [tokenizer.decode(sample, skip_specials_tokens=True)
                    for sample in generated_ids]
        
            re_key = '^answer.*?\n'

            bodies = [completion[:re.search(re_key, completion).span()[1]]
                if re.search(re_key, completion) else completion
                for completion in untrunced_bodies]

            answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]

            passed_lst = [(abs((answer - label.item())/label.item()) < 0.01) 
                    if isinstance(answer, float) else False 
                    for answer in answers]

            if True in passed_lst: 
                gold_code = bodies[passed_lst.index(True)]
                solved.add(idx.item())
                solutions[idx.item()] = gold_code

              
    return solved, solutions


                
def main(): 
    config_path = sys.argv[1] 

    with open(config_path, "r") as f: 
        cfg = yaml.safe_load(f)

    experiment_name = cfg['experiment_name']
    param_count = cfg['param_count']
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
    device = 'cuda'
    max_length = cfg['max_length']
    epochs_per_step = cfg['epochs']
    inference_batch_size = cfg['inference_batch_size']
    inference_num_samples = cfg['inference_num_samples']
    train_batch_size = cfg['train_batch_size']
    num_iters = cfg['num_iters']
    weight_decay = cfg['weight_decay']

    results_dir = f"results_train/{experiment_name}"
    os.mkdir(results_dir)

    model_name = f"EleutherAI/gpt-neo-{param_count}"
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Just doing 3000 examples 
    max_instances = 3000
    all_data_list = read_mathqapython('../data/mathqapython_train.json')
    random.shuffle(all_data_list)
    all_data_list = all_data_list[:max_instances]

    # Seed with 100 labelled examples 
    num_seeds = 1000
    solved = set(random.sample(range(max_instances), num_seeds))

    solutions = [None for _ in range(max_instances)]
    for i in solved: 
        solutions[i] = all_data_list[i].code 


    for i in range(num_iters): 
        labelled_examples = [change_code(all_data_list[i], solutions[i]) for i in solved]

        model = train_model(model, tokenizer, labelled_examples, f"MLE{i}", cfg)


        solved, solutions = update_solved(model, 
                                         tokenizer,
                                         solved, 
                                         solutions, 
                                         all_data_list, 
                                         temp=1, 
                                         cfg=cfg,
                                         )

        print("NUMBER SOLVED: ", len(solved))

        
        with open(f"results_train/{experiment_name}/S{i}.json", "w") as fle: 
            json.dump({"num solved": len(solved), 
                       "solutions": [{"idx": i, "prompt": all_data_list[i].text, "solution": solutions[i]} 
                                        for i in solved]
                      }, fle, indent=4)


if __name__=="__main__": 
    main()
