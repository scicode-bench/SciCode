import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from scicode.parse.parse import (
    extract_function_name,
    get_function_from_code,
    read_from_hf_dataset,
)
from scicode.gen.models import extract_python_script, get_model_function
from scicode.gen.token_records import preload_written_sample_ids

DEFAULT_PROMPT_TEMPLATE = Path("eval", "data", "background_comment_template.txt").read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("eval", "data", "multistep_template.txt").read_text()


class Gencode:
    def __init__(self, model: str, output_dir: Path,
                 prompt_dir: Path, with_background: bool, temperature: float,
                 api_key: str = "", base_url: str | None = None, max_tokens: int = 4096,
                 timeout: float = 3600.0, repetition_penalty: float | None = None,
                 stream: bool = False, api_type: str = "chat_completions",
                 extra_body: dict | None = None):
        self.model = model
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.repetition_penalty = repetition_penalty
        self.stream = stream
        self.api_type = api_type
        self.extra_body = extra_body
        self.previous_llm_code = []

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def save_prompt_with_steps(self, prob_data: dict, prompt: str, num_steps: int) -> None:
        output_dir = Path(self.prompt_dir, Path(self.model).parts[-1], self._get_background_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def save_response_with_steps(self, prob_data: dict, response: str,
                                 previous_code: str, num_steps: int) -> None:
        output_dir = (
                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")

    def save_step_generation_log(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        prompt: str,
        raw_response: str,
    ) -> None:
        log_dir = (
            self.output_dir
            / Path(self.model).parts[-1]
            / self._get_background_dir()
            / "generation_logs"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        log_path = log_dir / f"{prob_id}.{num_steps}.log"
        ts = datetime.now().isoformat(timespec="seconds")
        problem_name = prob_data.get("problem_name", "")
        desc_main = prob_data.get("problem_description_main", "")
        header = (
            f"problem_id={prob_id} step={num_steps}/{tot_steps} saved_at={ts}\n"
            f"problem_name={problem_name}\n"
        )
        overview = ""
        if desc_main:
            overview = f"\n=== Problem description (dataset) ===\n{desc_main}\n"
        body = (
            f"{header}{overview}\n"
            f"=== Prompt (sent to model) ===\n{prompt}\n\n"
            f"=== Response ===\n{raw_response}"
        )
        log_path.write_text(body, encoding="utf-8")

    def generate_response_with_steps(
        self, prob_data: dict, num_steps: int, tot_steps: int, model="gpt-4o",
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            *, save: bool = True) -> None:
        prob_id = prob_data["problem_id"]
        output_file_path = (
                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
                / f"{prob_id}.{num_steps}.py"
        )
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (prob_id == "13" and prev_step == 5) or (prob_id == "62" and prev_step == 0)\
                            or (prob_id == "76" and prev_step == 2):
                        prev_file_path = Path("eval", "data", f"{prob_id}.{prev_step+1}.txt")
                    else:
                        prev_file_path = (
                                self.output_dir / Path(self.model).parts[-1] / self._get_background_dir()
                                / f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        func_name = extract_function_name(prob_data["sub_steps"][prev_step]["function_header"])
                        function_code = get_function_from_code(prev_file_content, func_name)
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(f'Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}.')

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(prob_data, num_steps, prompt_template)
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps)

        model_fct = get_model_function(
            model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            repetition_penalty=self.repetition_penalty,
            stream=self.stream,
            api_type=self.api_type,
            extra_body=self.extra_body,
        )
        # sample_id 用于 reuse 场景下按 id 去重 token records；与落盘 .py 文件名对齐
        response_from_llm = model_fct(prompt, sample_id=f"{prob_id}.{num_steps}")
        self.save_step_generation_log(prob_data, num_steps, tot_steps, prompt, response_from_llm)
        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(prob_data, response_from_llm, previous_code, num_steps)

    @staticmethod
    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(self, problem_data: dict, num_steps: int):
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
        output_str = "\n\n".join(output_lines[:-1])
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str

    def generate_prompt_with_steps(self, prob_data: dict, num_steps: int,
                                   prompt_template=DEFAULT_PROMPT_TEMPLATE):
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(prob_data,
                                                                                         num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.getenv("MODEL_API_KEY", ""), help="API key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--split", type=str, default="test",
                        choices=["validation", "test"], help="Dataset split")
    parser.add_argument("--problem-id", type=str, default=None,
                        help="Run only the specified problem_id")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: eval_results/<timestamp>_<split>)")
    parser.add_argument("--prompt-dir", type=Path, default=Path("eval_results", "prompt"),
                        help="Prompt directory")
    parser.add_argument("--with-background", action="store_true",
                        help="Include problem background")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Repetition penalty (for vLLM / Bailian backends)")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming responses from the OpenAI-compatible API")
    parser.add_argument("--api-type", type=str, default="chat_completions",
                        choices=["chat_completions", "responses"],
                        help="OpenAI API type")
    parser.add_argument("--extra-body", type=str, default=None,
                        help="JSON object merged into the API request body")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers (one per problem)")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="HTTP timeout in seconds per API call")
    return parser


SKIP_STEPS = {("13", 5), ("62", 0), ("76", 2)}


def _generate_one_problem(problem: dict, gcode: Gencode, model: str,
                          prompt_template: str) -> str:
    prob_id = problem['problem_id']
    steps = len(problem['sub_steps'])
    print(f'Generating {prob_id}...')
    to_run = [(i, i + 1) for i in range(steps) if (prob_id, i) not in SKIP_STEPS]
    for _i, num_step in tqdm(to_run, desc=f"problem {prob_id}", unit="step", leave=False):
        gcode.generate_response_with_steps(problem, num_step, steps, model, prompt_template)
    return prob_id


def main(model: str,
         split: str,
         problem_id: str | None,
         output_dir: Path | None,
         prompt_dir: Path,
         with_background: bool,
         temperature: float,
         repetition_penalty: float | None,
         api_key: str = "",
         base_url: str | None = None,
         max_tokens: int = 4096,
         num_workers: int = 8,
         timeout: float = 3600.0,
         stream: bool = False,
         api_type: str = "chat_completions",
         extra_body: str | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("eval_results", f"{timestamp}_{split}")
    print(f'Output directory: {output_dir}')

    # 预加载 token_records 中已写入的 sample_id，防止 reuse 补跑时重复写入
    # ThreadPoolExecutor 全线程共享同一份 set，只需在此调用一次
    n_preloaded = preload_written_sample_ids()
    if n_preloaded > 0:
        print(f'Preloaded {n_preloaded} sample_ids from existing token_records (reuse dedup)')

    parsed_extra_body = json.loads(extra_body) if extra_body else None

    prompt_template = BACKGOUND_PROMPT_TEMPLATE if with_background else DEFAULT_PROMPT_TEMPLATE
    data = read_from_hf_dataset(split)
    if problem_id is not None:
        data = [problem for problem in data if problem["problem_id"] == problem_id]
        if not data:
            raise ValueError(f"Problem {problem_id} not found in split '{split}'.")
        print(f"Running only problem {problem_id}")

    def make_gcode():
        return Gencode(
            model=model, output_dir=output_dir,
            prompt_dir=prompt_dir, with_background=with_background, temperature=temperature,
            api_key=api_key, base_url=base_url, max_tokens=max_tokens,
            timeout=timeout, repetition_penalty=repetition_penalty,
            stream=stream, api_type=api_type, extra_body=parsed_extra_body,
        )

    with ThreadPoolExecutor(max_workers=min(num_workers, len(data))) as executor:
        futures = {
            executor.submit(_generate_one_problem, problem, make_gcode(), model, prompt_template): problem['problem_id']
            for problem in data
        }
        for future in as_completed(futures):
            prob_id = futures[future]
            try:
                future.result()
                print(f'Done {prob_id}')
            except Exception as e:
                print(f'Error on problem {prob_id}: {e}')


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
