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


def test_run_and_test_single_issue(tmpdir):
    cmd = [
        "python",
        "eval/scripts/gencode_json.py",
        "--model",
        "dummy",
        "--output-dir",
        str(tmpdir),
        "--prompt-dir",
        str(tmpdir / "prompts"),
        "--input-path",
        "tests/test_data/first_problem.jsonl",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    # this doesn't currently do anything because
    # gpt-4o is hardcoded as model
    cmd = [
        "python",
        "eval/scripts/test_generated_code.py",
    ]