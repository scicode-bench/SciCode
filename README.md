# SciCode 

[**Homepage**](https://scicode-bench.github.io/) | [**Paper**](https://arxiv.org/abs/2407.13168)


This repo contains the evaluation code for the paper "[SciCode: A Research Coding Benchmark Curated by Scientists](https://arxiv.org/abs/2407.13168)"

## Setup

```bash
git clone <this-repo>
cd SciCode
uv sync
source .venv/bin/activate
```

Download the numeric test data and place it at `eval/data/test_data.h5`:
- [Google Drive](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link)

The problem datasets (`problems_dev.jsonl`, `problems_test.jsonl`) are already included in `eval/data/`.

## Step 1 — Generate code

```bash
uv run python eval/scripts/gencode.py \
  --model <model-name> \
  --api-key <your-api-key> \
  --base-url <api-base-url> \
  --split test \
  --with-background \
  --output-dir eval_results/my_run \
  --num-workers 16 \
  --max-tokens 65536 \
  --timeout 600 \
  --repetition-penalty 1.1
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | `gpt-4o` | Model name passed to the API |
| `--api-key` | `""` | API key for the model service |
| `--base-url` | `None` | Base URL for OpenAI-compatible endpoints (e.g. Bailian, vLLM) |
| `--split` | `test` | Dataset split: `test` (65 problems) or `validation` (15 problems) |
| `--with-background` | off | Include scientist-annotated background in the prompt |
| `--output-dir` | auto timestamp | Where to save generated `.py` files; fixed name enables resume |
| `--num-workers` | `8` | Parallel threads (one per problem) |
| `--max-tokens` | `4096` | Max tokens per generation |
| `--timeout` | `180` | HTTP timeout per API call in seconds |
| `--repetition-penalty` | `None` | Repetition penalty (supported by vLLM / Bailian backends) |
| `--temperature` | `0` | Sampling temperature |
| `--problem-id` | `None` | Run a single problem by ID |

**Output structure:**

```
eval_results/my_run/
  <model-name>/
    with_background/
      1.1.py   # problem 1, step 1
      1.2.py   # problem 1, step 2
      ...
      generation_logs/
        1.1.log  # full prompt + raw response per step
  prompt/
    <model-name>/with_background/
      1.1.txt  # prompt sent to model
```

**Resume support:** re-run with the same `--output-dir` and already-generated files are skipped automatically.

**Example (Qwen via Bailian):**

```bash
uv run python eval/scripts/gencode.py \
  --model qwen3.5-122b-a10b \
  --api-key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --max-tokens 65536 \
  --split test \
  --with-background \
  --num-workers 16 \
  --repetition-penalty 1.1 \
  --timeout 600 \
  --output-dir eval_results/my_run_test
```

## Step 2 — Evaluate

```bash
bash evalcode.sh <generated_dir> [split]
```

`<generated_dir>` is the path ending in `with_background` or `without_background`, e.g.:

```bash
bash evalcode.sh eval_results/my_run_test/qwen3.5-122b-a10b/with_background test
```

Or call the script directly:

```bash
uv run python eval/scripts/test_generated_code.py \
  --generated-dir eval_results/my_run_test/qwen3.5-122b-a10b/with_background \
  --split test \
  --output-dir eval_results/my_run_test
```

**Eval output:**

```
eval_results/my_run_test/
  qwen3.5-122b-a10b_with_background.txt   # summary: correct problems / steps
  qwen3.5-122b-a10b_with_background.json  # per-problem pass/fail detail
  logs/
    qwen3.5-122b-a10b/with_background/
      1.1.txt  # "pass" / "fail" / "time out" per step
```

## Citation
```bibtex
@misc{tian2024scicode,
    title={SciCode: A Research Coding Benchmark Curated by Scientists},
    author={Minyang Tian and Luyu Gao and Shizhuo Dylan Zhang and Xinan Chen and Cunwei Fan and Xuefei Guo and Roland Haas and Pan Ji and Kittithat Krongchon and Yao Li and Shengyan Liu and Di Luo and Yutao Ma and Hao Tong and Kha Trinh and Chenyu Tian and Zihan Wang and Bohao Wu and Yanyu Xiong and Shengzhu Yin and Minhui Zhu and Kilian Lieret and Yanxin Lu and Genglin Liu and Yufeng Du and Tianhua Tao and Ofir Press and Jamie Callan and Eliu Huerta and Hao Peng},
    year={2024},
    eprint={2407.13168},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
