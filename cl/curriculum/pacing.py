import math 

class PacingFunction: 
    def get_fraction(self, iter_num): 
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

    def get_fraction(self, iter_num): 
        return min(1, start_prop * math.pow(increase, math.floor(iter_num / step_length)))


class ThreeLengthVariableExponential(PacingFunction): 
    def __init__(self, 
                 start_prop, 
                 step_lengths, 
                 increase
                ): 
        self.start_prop = start_prop 
        self.step_length = step_length 
        self.increase = increase 

    def get_fraction(self, iter_num): 
        if iter_num < step_lengths[0]: 
            return start_prop
        elif iter_num < step_lengths[1]: 
            return min(1, increase * start_prop)
        else: 
            return min(1, start_prop * math.pow(increase, math.floor((iter_num - step_lengths[0] - step_lengths[1])/step_lengths[2] + 2)))
