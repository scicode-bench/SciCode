import pytest
import subprocess


def test_smoke_test():
    subprocess.run(["python", "eval/scripts/gencode_json.py", "--help"], check=True)


def test_run_gencode_dummy_model(tmpdir):
    cmd = [
        "python",
        "eval/scripts/gencode_json.py",
        "--model",
        "dummy",
        "--output-dir",
        str(tmpdir),
        "--prompt-dir",
        str(tmpdir / "prompts"),
    ]
    subprocess.run(cmd, check=True)