from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

dataset = json_dataset(
    "/eagle/tpc/zilinghan/SciCode/eval/data/problems_dev.jsonl",
    FieldSpec(
        input="problem_description_main",
        target="problem_id",
        id="problem_id",
    ),
)

for i in len(dataset):
    print(dataset[i])