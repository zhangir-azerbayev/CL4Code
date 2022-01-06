import math 

class PacingFunction: 
    def __call__(self, iter_num): 
        pass

class FixedExponential(PacingFunction): 
    def __init__(self, 
                 start_prop, 
                 step_length, 
                 increase
                ):
        self.start_prop = start_prop
        self.step_length = step_length 
        self.increase = increase 

    def __call__(self, iter_num): 
        return min(1, self.start_prop * math.pow(self.increase, math.floor(iter_num / self.step_length)))


class ThreeLengthExponential(PacingFunction): 
    def __init__(self, 
                 start_prop, 
                 step_lengths, 
                 increase
                ): 
        self.start_prop = start_prop 
        self.step_lengths = step_lengths
        self.increase = increase 

    def __call__(self, iter_num): 
        if iter_num < self.step_lengths[0]: 
            return self.start_prop
        elif iter_num < self.step_lengths[0] + self.step_lengths[1]: 
            return min(1, self.increase * self.start_prop)
        else: 
            return min(1, self.start_prop * math.pow(self.increase, math.floor((iter_num - self.step_lengths[0] -
                self.step_lengths[1])/self.step_lengths[2] + 2)))
