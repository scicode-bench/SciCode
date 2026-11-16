"""验证原生样本重试、错误保留和计分边界，不访问模型服务。"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
from inspect_ai import eval as inspect_eval, eval_retry
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.log import read_eval_log


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src"))
    monkeypatch.chdir(root / "eval/inspect_ai")
    spec = importlib.util.spec_from_file_location("native_scicode_retry", "scicode.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def problem(identifier="1"):
    return {
        "problem_id": identifier,
        "required_dependencies": "",
        "sub_steps": [{
            "step_number": f"{identifier}.1",
            "function_header": "def answer():",
            "return_line": "    return value",
            "step_description_prompt": f"Return one for problem {identifier}.",
            "step_background": "",
            "test_cases": [],
        }],
    }


def test_opt_in_generation_error_stops_without_placeholder(native, tmp_path):
    from inspect_ai.model import ChatMessageUser, ModelName
    from inspect_ai.solver import TaskState

    state = TaskState(model=ModelName("mockllm/model"), sample_id="1", epoch=1,
                      input="problem", messages=[ChatMessageUser(content="problem")],
                      metadata=problem())
    calls = []

    async def fail(**kwargs):
        calls.append(1)
        raise TimeoutError("request failed")

    solve = native.scicode_solver(output_dir=str(tmp_path), with_background=False,
                                  mode="normal", stream=False, stop_on_generation_error=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(solve(state, fail))
    assert len(calls) == 1
    assert state.metadata["scicode_generation_errors"][0]["type"] == "TimeoutError"
    assert not list(tmp_path.rglob("*.py"))


def test_strict_task_stops_without_scoring_failed_sample(native, monkeypatch, tmp_path):
    record = problem()
    monkeypatch.setattr(native, "hf_dataset", lambda *a, **k: [
        Sample(id="1", input="problem", target="1", metadata=record)
    ])
    scored = []
    monkeypatch.setattr(native.ScicodeEvaluator, "test_code", lambda *args: scored.append(1))
    calls = []

    def output(*args):
        calls.append(1)
        raise TimeoutError("request failed")

    task = native.scicode(output_dir=str(tmp_path / "artifacts"), stream=False,
                          stop_on_generation_error=True)
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=output),
                       retry_on_error=0, max_retries=0, max_samples=1,
                       log_dir=str(tmp_path / "logs"), log_format="json", display="none")[0]
    assert len(calls) == 1
    assert not scored
    assert log.status == "error"


@pytest.mark.parametrize("stream", [True, False])
def test_solver_stream_option_controls_setup(native, monkeypatch, tmp_path, stream):
    from inspect_ai.model import ChatMessageUser, ModelName
    from inspect_ai.solver import TaskState

    record = problem()
    record["sub_steps"] = []
    state = TaskState(model=ModelName("mockllm/model"), sample_id="1", epoch=1,
                      input="problem", messages=[ChatMessageUser(content="problem")],
                      metadata=record)
    calls = []
    monkeypatch.setattr(native, "get_model", lambda: "model")
    monkeypatch.setattr(native, "enable_streaming", lambda model: calls.append(model))
    solve = native.scicode_solver(output_dir=str(tmp_path), with_background=False,
                                  mode="normal", stream=stream)
    asyncio.run(solve(state, None))
    assert calls == (["model"] if stream else [])
