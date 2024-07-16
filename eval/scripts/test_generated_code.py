from pathlib import Path
import os
import json
import subprocess
import time
import numpy as np

from scicode.parse.parse import read_from_jsonl


prob_num = 65
step_num = 288

logs_dir = 'eval/logs'
code_dir = 'eval_results / generated_code'
test_result_dir = 'test_result'


if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

json_path = 'eval/data/problems_all.jsonl'
json_dct = {}
json_idx = {}

jsonl_data = read_from_jsonl(json_path)
for prob_data in jsonl_data:
    json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
    json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)


def test_code(model_name):
    start_time = time.time()

    code_dir_ = f'{code_dir}/{model_name}'
    tmp_dir = f'tmp_{start_time}'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for root, _, files in os.walk(code_dir_):
        for file in files:
            file_name = Path(file).stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = Path(root, file).read_text(encoding='utf-8')
            json_content = jsonl_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')

    def run_script(script_path):
        try:
            subprocess.run(['python', script_path], check=True, capture_output=True,
                           text=True, timeout=1800)
            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error running script {script_path}: {e}")
            print(e.output)
            return 1
        except subprocess.TimeoutExpired as e:
            print(f"Runtime error while running script {script_path}: {e}")
            return 2

    correct_prob = np.zeros(prob_num)
    tot_prob = np.zeros(prob_num)
    correct_step = []
    correct_dict = {}

    for i in range(prob_num):
        correct_dict[f'{i+1}'] = []

    for root, _, files in os.walk(tmp_dir):
        for file in files:
            script_path = Path(root, file)
            func_id = str(file.split('.py')[0])
            prob_id = str(func_id.split('.')[0])
            print(f'Testing function {func_id} ...')
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = f'{logs_dir}/{model_name}'
            if not os.path.exists(logs_dir_):
                os.makedirs(logs_dir_)
            logs_file = os.path.join(logs_dir_, f'{Path(file).stem}.txt')
            if os.path.exists(logs_file):
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        correct_prob[int(prob_id) - 1] += 1
                        correct_step.append(func_id)
                        correct_dict[str(prob_id)].append(func_id)
                continue
            ret = run_script(script_path)
            if ret == 0:
                correct_prob[int(prob_id) - 1] += 1
                correct_step.append(func_id)
                correct_dict[str(prob_id)].append(func_id)
                with open(logs_file, 'w') as f:
                    f.write('pass')
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')

    test_time = time.time() - start_time

    correct_prob_num = sum(1 for i in range(prob_num) if
                           correct_prob[i] == tot_prob[i]
                           and tot_prob[i] != 0)

    print(f'correct problems: {correct_prob_num}/{prob_num}')
    print(f'correct steps: {len(correct_step)}/{step_num}')

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    with open(f'{test_result_dir}/{model_name}.txt', 'w') as f:
        f.write(f'correct problems(include dev set): {correct_prob_num}/{prob_num}\n')
        f.write(f'correct steps(include dev set): {len(correct_step)}/{step_num}\n\n')
        f.write(f'duration: {test_time} seconds\n')
        f.write('\ncorrect problems: ')
        f.write(f'\n\n{[i + 1 for i in range(prob_num) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0]}\n')

    with open(f'{test_result_dir}/{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(correct_dict, f, indent=4)
    
    Path(tmp_dir).rmdir()




model = 'gpt-4o'
test_code(model)
