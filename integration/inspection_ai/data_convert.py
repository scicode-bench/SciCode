"""
This is a script to convert the original jsonl formated SciCode data to a json formatted for using inspection_ai to evaluate models.
"""

import copy
import json
from pathlib import Path
from scicode.parse.parse import read_from_jsonl

input_paths = [
    '../../eval/data/problems_dev.jsonl',
    '../../eval/data/problems_all.jsonl',
]

output_paths = [
    'data/problems_dev_new.json',
    'data/problems_all_new.json',
]

# for input, output in zip(input_paths, output_paths):
#     data = read_from_jsonl(input)
#     data_converted = []
#     for problem in data:
#         problem_converted = copy.deepcopy(problem)
#         for sub_step in problem['sub_steps']:
#             problem_converted = copy.deepcopy(problem)
#             for key, value in sub_step.items():
#                 problem_converted[key] = value
#             problem_converted["problem_id"] = problem_converted["step_number"]
#             data_converted.append(problem_converted)
    
#     with open(output, 'w') as f:
#         json.dump(data_converted, f, indent=4)

for input, output in zip(input_paths, output_paths):
    data = read_from_jsonl(input)
    
    with open(output, 'w') as f:
        json.dump(data, f, indent=4)
