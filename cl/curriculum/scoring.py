from cl.data.dataset import MathQAInstance 
from transformers import GPT2Tokenizer

class ScoringFunction: 
    def score(self, instance: MathQAInstance): 
        pass


class CodeLength(ScoringFunction): 
    def __init__(self): 
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    def __call__(self, instance: MathQAInstance): 
        return len(self.tokenizer.encode(instance.code))

class SequenceLength(ScoringFunction): 
    def __init__(self): 
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    def score(self, instance: MathQAInstance): 
        return len(self.tokenizer.encode(instance.text + '\n' + instance.code))

class SelfTaught(ScoringFunction): 
    def __init__(self, model, param_count, device): 
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")
        self.model = model 
        self.device = device

    def score(self, instance: MathQAInstance): 
        encodings_dict = self.tokenizer(instance.code + '\n' + instance.text, 
                return_tensors='pt').to(self.device)
        output_dict = model.forward(**encodings_dict) 

        return output_dict["loss"]



