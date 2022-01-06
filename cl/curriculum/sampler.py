import math 
import random 
from torch.utils.data import Sampler, BatchSampler


class _QueueIndexIter: 
    def __init__(self, max_idx): 
        self.max_idx = max_idx
        self.q = list(range(max_idx))

    def __iter__(self): 
        while True: 
            if not self.q: 
                self.q = list(range(self.max_idx))

            idx = random.randrange(len(self.q))

            temp = self.q[idx]
            self.q[idx] = self.q[-1]
            self.q[-1] = temp 

            yield self.q.pop()

    def update_max_idx(self, max_idx): 
        self.q = self.q + list(range(self.max_idx, max_idx))

        self.max_idx = max_idx 


class CurriculumSampler(Sampler): 
    def __init__(self, pacer, dataset, batch_size, epochs): 
        self.pacer = pacer
        self.cumulative_iters = 0 
        self.num_instances = len(dataset)
        self.batch_size = batch_size 
        self.epochs = epochs

    def __iter__(self): 
        for _ in range(math.ceil(self.num_instances/self.batch_size) * self.epochs):
            max_idx = round(self.num_instances * self.pacer(self.cumulative_iters))
            for sample in BatchSampler(range(max_idx), self.batch_size, drop_last=False): 
                batch = sample 
                break 
            yield batch 
            self.cumulative_iters += 1 
    def __len__(self): 
        return math.ceil(self.num_instances/self.batch_size) * self.epochs




