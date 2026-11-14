"""Check scoring through Inspect, generated programs, and numerical references."""

import importlib.util
from pathlib import Path
import sys

import h5py
import pytest
from inspect_ai import Task, eval


@pytest.mark.parametrize("incorrect", [False, True])
def test_reference_steps_are_not_scored(tmp_path, monkeypatch, incorrect):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "eval" / "inspect_ai")
    spec = importlib.util.spec_from_file_location("scicode_eval", "scicode.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    code = tmp_path / "mockllm-fixture" / "generated_code" / "without_background"
    code.mkdir(parents=True)
    reference = tmp_path / "reference.h5"
    samples = []
    expected_scores = {}
    # The supplied steps are part of the question, as documented in #1 and #12.
    cases = [("13", 7, 6), ("62", 2, 1), ("76", 4, 3), ("77", 2, None)]
    with h5py.File(reference, "w") as handle:
        for problem_id, step_count, supplied_step in cases:
            steps = []
            wrong_step = next(
                index for index in range(1, step_count + 1) if index != supplied_step
            )
            for index in range(1, step_count + 1):
                step = f"{problem_id}.{index}"
                steps.append(
                    {
                        "step_number": step,
                        "test_cases": [
                            "if solve() != target:\n    raise ValueError('wrong result')"
                        ],
                    }
                )
                if index == supplied_step:
                    continue
                handle.create_dataset(f"{step}/test1/var1", data=7)
                value = 8 if incorrect and index == wrong_step else 7
                (code / f"{step}.py").write_text(f"def solve():\n    return {value}\n")
            samples.append(
                module.record_to_sample({"problem_id": problem_id, "sub_steps": steps})
            )
            evaluated = step_count - (supplied_step is not None)
            expected_scores[problem_id] = {
                "Problem Correctness": int(not incorrect),
                "Total Correct": evaluated - incorrect,
                "Total Steps": evaluated,
            }

    for run in ("fresh", "cached"):
        task = Task(
            dataset=samples,
            solver=[],
            scorer=module.scicode_scorer(
                output_dir=tmp_path,
                with_background=False,
                h5py_file=str(reference),
            ),
        )
        log = eval(
            task,
            model="mockllm/fixture",
            display="none",
            log_dir=str(tmp_path / "inspect_logs"),
            max_samples=1,
        )[0]
        if log.status != "success":
            pytest.fail(f"{run} evaluation failed: {log.error}")
        for sample in log.samples:
            actual = next(iter(sample.scores.values())).value
            if actual != expected_scores[sample.id]:
                pytest.fail(f"{run} problem {sample.id}: {actual}")
        metrics = {
            name: metric.value
            for score in log.results.scores
            for name, metric in score.metrics.items()
        }
        expected_metrics = {
            "mean": float(not incorrect),
            "sub_problem_correctness": (12 - 4 * incorrect) / 12,
        }
        if metrics != expected_metrics:
            pytest.fail(f"{run} metrics {metrics} differ from {expected_metrics}")
