import subprocess


def test_smoke_test():
    subprocess.run(["python", "eval/scripts/gencode.py", "--help"], check=True)


def test_run_gencode_dummy_model(tmpdir):
    cmd = [
        "python",
        "eval/scripts/gencode.py",
        "--model",
        "dummy",
        "--output-dir",
        str(tmpdir),
        "--prompt-dir",
        str(tmpdir / "prompts"),
        "--split",
        "validation",
    ]
    subprocess.run(cmd, check=True)
