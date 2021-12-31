from trax import data
import json
import numpy as np
import os
import tensorflow as tf



dataset_path = 'downloads/'



mathqa_test_gen = data.CreateMathQAInputs(dataset_path=dataset_path, cumulative=False, python_code=True, full_dict=True, train=False, test=True)()
mathqa_train_gen = data.CreateMathQAInputs(dataset_path=dataset_path, cumulative=False, python_code=True,full_dict=True, train=True, test=False)()
mathqa_dev_gen = data.CreateMathQAInputs(dataset_path=dataset_path, cumulative=False, python_code=True,full_dict=True, train=False, test=False)()

def read_all_problems(mathqa_gen):
  problems = []
  questions = set()
  index = 0
  while True:
    problem = next(mathqa_gen)
    problem_dict = {}
    if problem[0] in questions:
      break
    else:
      problem_dict['text'] = problem[0]
      problem_dict['code'] = problem[1]
      problem_dict['dsl_code'] = problem[2]
      problem_dict['reasoning'] = problem[3].strip('\"').strip("\'")
      problem_dict['answer'] = data.tf_inputs.execute_mathqa_program(problem[0], problem[1].split('\n'))
      problem_dict['task_id'] = index
      np.testing.assert_almost_equal(problem_dict['answer'], data.tf_inputs.execute_mathqa_dsl_program(problem[0], [problem[2]]))
      problems.append(problem_dict)
      questions.add(problem[0])
      index += 1
  return problems


train_problems = read_all_problems(mathqa_train_gen)
test_problems = read_all_problems(mathqa_test_gen)
dev_problems = read_all_problems(mathqa_dev_gen)

with open('mathqapython_train.json', 'w') as f: 
    json.dump(train_problems, f)

with open('mathqapython_test.json', 'w') as f: 
    json.dump(test_problems, f)

with open('mathqapython_dev.json', 'w') as f: 
    json.dump(dev_problems, f)

print(test_problems[0])
print(test_problems[1])

