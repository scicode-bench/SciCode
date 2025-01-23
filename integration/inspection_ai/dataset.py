import os
import time
from typing import Any
from pathlib import Path
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message
from scicode.parse.parse import (
    extract_function_name,
    get_function_from_code,
    read_from_jsonl
)
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    AnswerPattern,
    Score,
    Target,
    accuracy,
    stderr,
    scorer,
)
from scicode.gen.models import generate_dummy_response, extract_python_script

DEFAULT_PROMPT_TEMPLATE = Path("data", "background_comment_template.txt").read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("data", "multistep_template.txt").read_text()

# SCICODE_DATA_JSON_PATH = "/eagle/tpc/zilinghan/SciCode/integration/inspection_ai/data/problems_dev.json"
# SCICODE_DATA_JSON_PATH = "/eagle/tpc/zilinghan/SciCode/integration/inspection_ai/data/problems_all.json
SCICODE_DATA_JSON_PATH = "/eagle/tpc/zilinghan/SciCode/integration/inspection_ai/data/problems_dev_new.json"
TEMP_DIR = "./tmp"
MODEL_NAME = "gpt-4o"
WITH_BACKGROUND = False
TIMEOUT = 1000
SAVE = True

class PromptingAssistant:
    def __init__(
        self,
        output_dir: Path,
        prompt_dir: Path,
        with_background: bool,
    ):
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.previous_llm_code = []
        
    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"
    
    def register_previous_response(
        self,
        prob_data: dict,
        response: str,
        num_steps: int,
    ):
        self.previous_llm_code[num_steps - 1] = response
    
    def save_response_with_steps(
        self, 
        prob_data: dict, 
        response: str,
        previous_code: str, 
        num_steps: int
    ) -> None:
        output_dir = Path(
            self.output_dir,
            self._get_background_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")    
    
    @staticmethod
    def process_problem_code(
        prob_data: dict, 
        num_steps: int
    ) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string
    
    def process_problem_steps(
        self, 
        problem_data: dict, 
        num_steps: int
    ):
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"] if self.with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            output_lines.append(self.previous_llm_code[i])
            previous_code.append(self.previous_llm_code[i])
            output_lines.append("------")

        next_step.append(problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"] if self.with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(self.process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str
    
    def generate_prompt_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    ):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(prob_data, num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'
    
    def save_prompt_with_steps(
            self, 
            prob_data: dict, 
            prompt: str, 
            num_steps: int
        ) -> None:
        output_dir = Path(self.prompt_dir, self._get_background_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def prepare_final_prompt_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        *,
        save: bool = True
    ):
        prob_id = prob_data["problem_id"]
        output_file_path = Path(
            self.output_dir, 
            self._get_background_dir(),
            f"{prob_id}.{num_steps}.py"
        )
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5) or 
                        (prob_id == "62" and prev_step == 0) or 
                        (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = os.path.join("data", f"{prob_id}.{prev_step+1}.txt")
                    else:
                        prev_file_path = Path(
                            self.output_dir,
                            self._get_background_dir(),
                            f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code
                        print(f'Loaded previous code for problem {prob_id} step {prev_step + 1}')
                    else:
                        raise Exception(f'Generating problem {prob_id} step {num_steps} ahead of step {prev_step + 1}.')
                
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data,
            num_steps,
            prompt_template,
        )
        if save:
            self.save_prompt_with_steps(
                prob_data,
                prompt,
                num_steps,
            )
        return prompt, previous_code
            

def save_prompt_with_steps(
    record,
    prompt,
    num_steps,
):
    output_dir = Path(
        TEMP_DIR,
        "prompt",
        MODEL_NAME,
        "with_background" if WITH_BACKGROUND else "without_background"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{record['problem_id']}.{num_steps}.txt"
    output_file_path.write_text(prompt, encoding="utf-8")

def process_problem_code(
    record,
    prob_step_id,
):
    header_docstring = record["sub_steps"][prob_step_id-1]["function_header"]
    return_str = record["sub_steps"][prob_step_id-1]["return_line"]
    return f"{header_docstring}\n\n{return_str}"

def process_problem_steps(record, prob_step_id, previous_llm_code):
    output_lines = []
    next_step = []
    previous_code = []
    for i in range(prob_step_id-1):
        output_lines.append(
            record["sub_steps"][i]["step_description_prompt"] + '\n' +
            record["sub_steps"][i]["step_background"] if WITH_BACKGROUND else record["sub_steps"][i]["step_description_prompt"]
        ) # TODO: Check if this is correct
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")
    
    next_step.append(
        record["sub_steps"][prob_step_id-1]["step_description_prompt"] + '\n' +
        record["sub_steps"][prob_step_id-1]["step_background"] if WITH_BACKGROUND else record["sub_steps"][prob_step_id-1]["step_description_prompt"]
    )
    next_step.append(
        process_problem_code(record, prob_step_id)
    )
    output_str = "\n\n".join(output_lines[:-1])
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return output_str, next_step_str, previous_code_str
    
def generate_prompt_with_steps(
    record,
    prob_step_id,
    prompt_template,
    previous_llm_code,
):
    problem_step_str, next_step_str, previous_code_str = process_problem_steps(record, prob_step_id, previous_llm_code)
    depedencies = record["required_dependencies"]
    assert next_step_str
    return prompt_template.format(
        problem_steps_str=problem_step_str,
        next_step_str=next_step_str,
        dependencies=depedencies,
    ), f'{depedencies}\n{previous_code_str}\n'

# Dataset preparation
def record_to_sample_fake(record):
    prob_id = record["problem_id"]
    output_file_path = os.path.join(
        TEMP_DIR,
        "generated_code",
        MODEL_NAME,
        "with_background" if WITH_BACKGROUND else "without_background",
        f"{prob_id}.py"
    )
    
    prob_main_id, prob_step_id = prob_id.split(".")
    prob_step_id = int(prob_step_id)
    previous_llm_code = []
    if prob_step_id != 1:
        for prev_step in range(prob_step_id - 1):
            if (prob_main_id == "13" and prev_step == 5) or (prob_main_id == "62" and prev_step == 0) or (prob_main_id == "76" and prev_step == 2):
                prev_file_path = os.path.join("data", f"{prob_main_id}.{prev_step+1}.txt")
                exist_flag = True
            else:
                prev_file_path = os.path.join(
                    TEMP_DIR,
                    "generated_code",
                    MODEL_NAME,
                    "with_background" if WITH_BACKGROUND else "without_background",
                    f"{prob_main_id}.{prev_step + 1}.py"
                )
                prev_exist_flag_file_path = os.path.join(
                    TEMP_DIR,
                    "generated_code",
                    MODEL_NAME,
                    "with_background" if WITH_BACKGROUND else "without_background",
                    f"_{prob_main_id}.{prev_step + 1}.txt"
                )
                exist_flag = False
            # Wait until the previous file is generated
            timer = 0
            while (
                (not exist_flag) and 
                (not os.path.exists(prev_exist_flag_file_path))
            ):
                time.sleep(1)
                timer += 1
                if timer > TIMEOUT:
                    raise TimeoutError(f"Timeout waiting for {prev_exist_flag_file_path}")
            # Read previous code
            prev_file_content = Path(prev_file_path).read_text(encoding='utf-8')
            func_name = extract_function_name(
                record["sub_steps"][prev_step]["function_header"]
            )
            function_code = get_function_from_code(
                prev_file_content, func_name
            )
            previous_llm_code.append(function_code)
            
    prompt, previous_code = generate_prompt_with_steps(
        record,
        prob_step_id,
        DEFAULT_PROMPT_TEMPLATE if not WITH_BACKGROUND else BACKGOUND_PROMPT_TEMPLATE,
        previous_llm_code,
    )
    
    if SAVE:
        save_prompt_with_steps(
            record,
            prompt,
            prob_step_id,
        )
    
    record["_output_file_path"] = output_file_path
    record["_previous_code"] = previous_code
    print(f'Generated prompt for problem {prob_id}')

    return Sample(
        input=prompt,
        target=record["problem_id"],
        id=record["problem_id"],
        metadata={
            k: v for k, v in record.items() if k not in ["problem_id"]
        }
    )

def record_to_sample(record):
    return Sample(
        input="problem_id",
        target=record["problem_id"],
        id=record["problem_id"],
        metadata={
            k: v for k, v in record.items()
        }
    )

dataset = json_dataset(
    SCICODE_DATA_JSON_PATH, 
    record_to_sample
)
    

@solver
def dummy_solver(**params: dict[str, Any]):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt_assistant = PromptingAssistant(
            output_dir=Path(TEMP_DIR, "generated_code"),
            prompt_dir=Path(TEMP_DIR, "prompt"),
            with_background=WITH_BACKGROUND,
        )
        prompt_template = BACKGOUND_PROMPT_TEMPLATE if WITH_BACKGROUND else DEFAULT_PROMPT_TEMPLATE
        print('===============================')
        print(f'Processing problem {state.sample_id}')
        sub_steps = state.metadata["sub_steps"]
        for idx, step in enumerate(sub_steps):
            prompt, previous_code = prompt_assistant.prepare_final_prompt_with_steps(
                prob_data=state.metadata,
                num_steps=idx+1,
                tot_steps=len(sub_steps),
                prompt_template=prompt_template,
            )
            response_from_llm = generate_dummy_response(prompt)
            prompt_assistant.register_previous_response(
                prob_data=state.metadata,
                response=extract_python_script(response_from_llm),
                num_steps=idx+1,
            )
            
        print('===============================')
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def dummy_scorer():
    async def score(state: TaskState, target: Target):
        return Score(
            value=CORRECT
        )

    return score

@task
def dummy_task():
    return Task(
        dataset=dataset,
        solver=dummy_solver(),
        scorer=dummy_scorer(),
    )
