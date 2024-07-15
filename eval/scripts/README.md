 ## **Generate LLM code**
  
To run the script, go to the root of this repo and use the following command from the repository root:

```bash
python evaluation/scripts/gencode_json.py [options]
```

For example, to create  model results with `gpt-4o` and the default settings, run 

```bash
python evaluation/scripts/gencode_json.py --model gpt-4o
```

### Command-Line Arguments

- `--model` - Specifies the model name used for generating responses.
- `--output-dir` - Directory to store the generated code outputs (Default: `eval_results/generated_code`).
- `--input-path` - Directory containing the JSON files describing the problems (Default: `eval/data/problems_all.jsonl`).
- `--prompt-dir` - Directory where prompt files are saved (Default: `eval_results/prompt`).
- `--temperature` - Controls the randomness of the generation (Default: 0).
  
## **Evaluate generated code**

Download the [numeric test results](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) and save them as `./eval/data/test_data.h5`

To run the script, go to the root of this repo and use the following command:

```bash
python evaluation/scripts/test_generated_code.py
```

Please edit the `test_generated_code.py` source file to specify your model name, results directory and problem set (if not `problems_all.jsonl`).