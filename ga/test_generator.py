from ambiegen.testers.abstract_evolutionary_tester_ask_tell import AbstractEvolutionaryTesterAskTell
from ambiegen.generators.abstract_generator import AbstractGenerator

import numpy as np
import yaml
import random
class GATester(AbstractEvolutionaryTesterAskTell):
    def __init__(self, name, config):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(name, config)

    def initialize_test_generator(self):
        self.generator = DynamicTestGenerator(
            name="DynamicTestGenerator",
            size=40,
            min_val=0,
            max_val=4
        )

    def initialize_test_executors(self):
        self.executors = []

    
class GAInitTester(AbstractEvolutionaryTesterAskTell):
    def __init__(self, name, config, dynamic_tests = None):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(name, config)
        assert dynamic_tests is not None, "Dynamic tests must be provided for GAInitTester"
        self.dynamic_tests = dynamic_tests  



    def initialize_test_generator(self):

        dynamic_tests_num = len(self.dynamic_tests)
        # providing minimal values for such parameters
        #ego_x, ego_y, ego_heading, ego_speed, ego_target_lane
        #adv_x, adv_y, adv_heading, adv_speed, adv_target_lane, test index
        min_vals = [230, 0, -0.07, 21, 0, 280, 0, -0.07, 21, 0, 0]
        max_vals = [280, 1, 0.07, 29, 1, 340, 1, 0.07, 29, 1, dynamic_tests_num - 1]
        val_types = [float, int, float, float, int, float, int, float, float, int, int]
        self.generator = DynamicInitTestGenerator(
            name="DynamicTestGenerator",
            size=len(min_vals),
            min_val=min_vals,
            max_val=max_vals,
            val_types=val_types,
            dynamic_tests=self.dynamic_tests
        )

    def initialize_test_executors(self):
        self.executors = []

class DynamicTestGenerator(AbstractGenerator):
    def __init__(self, name, size, min_val, max_val):
        super().__init__(name)
        self._size = size
        self._min_val = min_val
        self._max_val = max_val

    def generate_random_test(self):
        # values from 0 to 4 inclusive
        values = np.arange(self._min_val, self._max_val + 1)

        # generate a random vector of size 40
        test = np.random.choice(values, size=self.size, replace=True)
        test = self.normalize(test, self._min_val, self._max_val)

        return test 
    

    def normalize(self, vec, min_val=0, max_val=4):
        """Normalize integer vector to range [0, 1]."""
        return (np.array(vec) - min_val) / (max_val - min_val)

    def denormalize(self, vec, min_val=0, max_val=4):
        """Denormalize [0, 1] vector back to integers [min_val, max_val]."""
        return np.round(vec * (max_val - min_val) + min_val).astype(int)

    @property
    def size(self):
        return self._size
    
    @property
    def lower_bound(self):
        return [0] * self.size

    @property
    def upper_bound(self):
        return [1] * self.size

    def genotype2phenotype(self, genotype):
        phenotype = self.denormalize(genotype, self._min_val, self._max_val)
        return phenotype

    def cmp_func(self, test1, test2):
        return np.linalg.norm(test1 - test2)

    def is_valid(self, test):
        return True, "Valid test"

    def visualize_test(self, test, save_path = None):
        pass



class DynamicInitTestGenerator(DynamicTestGenerator):
    def __init__(self, name, size, min_val, max_val, val_types, dynamic_tests):
        super().__init__(name, size, min_val, max_val)
        self._val_types = val_types
        self._dynamic_tests = dynamic_tests

    def normalize(self, vec):
        normed = []
        ranges = zip(self._min_val, self._max_val, self._val_types)
        for val, (low, high, dtype) in zip(vec, ranges):
            normed_val = (val - low) / (high - low)
            normed.append(normed_val)
        return normed


    def denormalize(self, vec):
        vector = []
        ranges = zip(self._min_val, self._max_val, self._val_types)
        for norm_val, (low, high, dtype) in zip(vec, ranges):
            val = low + norm_val * (high - low)
            if dtype == "int":
                val = int(round(val))  # round back to nearest integer
            vector.append(val)
        return vector

    def generate_random_test(self):
        vector = []
        ranges = zip(self._min_val, self._max_val, self._val_types)
        for (low, high, dtype) in ranges:
            if dtype == int:
                value = random.randint(low, high)
            elif dtype == float:
                value = random.uniform(low, high)
            else:
                raise ValueError("dtype must be 'int' or 'float'")
            vector.append(value)

        normalized_test = self.normalize(vector)
        return normalized_test
    
    def genotype2phenotype(self, genotype):
        result = {"init": None, "adv_actions": None}
        phenotype = self.denormalize(genotype)


        adv_actions = self._dynamic_tests[str(int(phenotype[-1]))]["adv_actions"]

        if phenotype[1] > 0.5:
            phenotype[1] = 4
        if phenotype[6] > 0.5:
            phenotype[6] = 4


        phenotype[4] =  ("0", "1", int(phenotype[4]))
        phenotype[9] =  ("0", "1", int(phenotype[9]))


        result["init"] = phenotype[:10]
        result["adv_actions"] = adv_actions

        return result
    

