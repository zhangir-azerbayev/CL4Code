import sys 
import os
import yaml 
import math

import torch

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from cl.data.dataset import read_mathqapython, MathQAPython

from train_expert_iter import update_solved, tokens_to_programs



def check_postprocessor(model, tokenizer): 
    prompts = ["to be or not to be that is", "I am on", "the montagne"]

    max_input_length=10

    encoded_texts = tokenizer(prompts, return_tensors="pt", 
            padding='max_length', max_length=max_input_length, 
            truncation=True)
    print("encoded texts: ", encoded_texts)

    num_samples = 3 

    print("sampling from model")
    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=1, 
                             max_new_tokens = 5, 
                             pad_token_id=tokenizer.eos_token_id, 
                             num_return_sequences = num_samples, 
                             )

    print("raw outputs: ", outputs)
    outputs = torch.reshape(outputs, (len(prompts), num_samples, -1))
    print("reshaped outputs: ", outputs)
    
    for out in outputs: 
        programs = tokens_to_programs(out, max_input_length, tokenizer)

        print(programs)


def check_generation(): 
    """
    texts = ["hello my name is", "you don't know my name!", "yes I do!"]

    encoded_texts = tokenizer(texts, return_tensors="pt",
            padding=True).to(device)

    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=0.2, 
                             max_new_tokens = 5, 
                             pad_token_id=tokenizer.eos_token_id, 
                             num_return_sequences = 2,
                             )


    print(encoded_texts)
    print(torch.reshape(outputs, (3, 2, -1)))
    """

    all_data_list = read_mathqapython('../data/mathqapython_train.json')[:15]

    solved = [0, 1, 13, 14] 
    solutions = [None for _ in all_data_list]

    print("#"*30)
    solved_solutions = update_solved(model, 
                        tokenizer, 
                        solved, 
                        solutions, 
                        all_data_list, 
                        5, 
                        2, 
                        .2, 
                        device,
                        )


    print(solved_solutions)


def main(): 
    tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-125M")

    check_postprocessor(model, tokenizer)

if __name__=="__main__": 
    main()
