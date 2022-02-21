import sys 
import os
import yaml 
import math

from itertools import product as product 

import torch 
import torch.nn 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import get_parameter_names

from data.dataset import read_mathqapython, MathQAPython

from cl.curriculum.pacing import ThreeLengthExponential
from cl.curriculum.scoring import CodeLength
from cl.curriculum.sampler import CurriculumSampler
from overrides import overrides

class CurriculumTrainer(Trainer): 
    def __init__(self, sampler, **kwargs): 
        super().__init__(**kwargs)
        self.sampler = sampler 
 
    @overrides
    def _get_train_dataloader(self) -> DataLoader: 
        return DataLoader(
                self.train_dataset, 
                collate_fn = self.data_collator, 
                num_workers = self.args.dataloader_num_workers, 
                pin_memory = self.args.dataloader_pin_memory, 
                batch_sampler = self.sampler
                )

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]), 
            'labels': torch.stack([f[0] for f in data])
           }

config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
param_count = cfg['param_count']
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
max_length = cfg['max_length']
epochs = cfg['epochs']
batch_size = cfg['batch_size']
#Optimizer 
optim = cfg['optimizer']
lr = optim['lr']
weight_decay = optim['weight_decay']
scheduler_type = optim['scheduler_type']
scheduler_kwargs = optim['scheduler_kwargs']
#Curriculum
curriculum = cfg['curriculum'] 
start_props = curriculum['start_props']
step_lengths = curriculum['step_lengths']
increases = curriculum['increases']
scoring_function_name = curriculum['scoring_function']

os.mkdir(f"results_train/{experiment_name}/")

print('loading data and configuring tokenizer')
data = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")
tokenizer.pad_token = tokenizer.eos_token 
# Creates sorted dataset 
if scoring_function_name == "CodeLength": 
    scoring_fn = CodeLength()
    sorted_data = sorted(data, key=scoring_fn)
    train_set = MathQAPython(sorted_data, tokenizer, max_length)
else: 
    raise ValueError("invalid scoring function")

steps_per_epoch = math.ceil(len(train_set)/batch_size)
print('loading model')
model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")

print('initializing training')
# Setting up optimizer 
# Parameter stuff is copied from huggingface 
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




with open(f"./results_train/{experiment_name}/config.yml", "w") as f: 
    yaml.dump(cfg, f)


for start_prop, step_1, step_2, step_3, increase in product(start_props, 
        step_lengths, step_lengths, step_lengths, increases): 
    if step_1 <= step_2 and step_2 <= step_3: 
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        if scheduler_type=="exponential": 
            gamma = scheduler_kwargs["gamma"]
            steps_per_epoch = math.ceil(len(train_set)/batch_size)
            lr_lambda = lambda step: gamma ** (step//steps_per_epoch)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else: 
            raise ValueError("invalid scheduler type")


        run_name = "_".join([str(x) for x in [start_prop, step_1, step_2, step_3, increase]])
        print(run_name)
        tb_writer = SummaryWriter(
                log_dir=f"./results_train/{experiment_name}/tb_log/{run_name}"
                )
        tb_callback = TensorBoardCallback(tb_writer)

        output_dir = f"./results_train/{experiment_name}/{run_name}"

        pacer = ThreeLengthExponential(start_prop, [step_1, step_2, step_3], increase)
        sampler = CurriculumSampler(pacer, train_set, batch_size, epochs)

        training_args = TrainingArguments(
                                          output_dir=output_dir,
                                          num_train_epochs=1, 
                                          logging_steps=steps_per_epoch,
                                          save_steps=10*steps_per_epoch,
                                          warmup_steps = 100, 
                                          )

        CurriculumTrainer(sampler, model=model, args=training_args, 
                train_dataset=train_set, data_collator=data_collator, 
                callbacks=[tb_callback], optimizers = (optimizer, scheduler)).train()
        tb_writer.close()

