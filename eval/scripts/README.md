 ## **Generate LLM code**
  
Your first need to set up your API keys. For this, create a `keys.cfg` file at the root of the repository
and add keys as follows:

```
OPENAI_KEY = 'your_api_key'
ANTHROPIC_KEY = 'your_api_key'
GOOGLE_KEY = 'your_api_key' 
```

For example, to create  model results with `gpt-4o` and the default settings, go to the root of this repo and run 

```bash
python eval/scripts/gencode_json.py --model gpt-4o
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
python eval/scripts/test_generated_code.py --model "model_name"
```
