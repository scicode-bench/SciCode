 - ## **Generate LLM code**
   
   To run the script, go to the root of this repo and use the following command:
   
   ```bash
   python evaluation/scripts/gencode_json.py [options]
   ```

   ### Command-Line Arguments
   - `--model` - Specifies the model name used for generating responses.
   - `--output-dir` - Directory to store the generated code outputs (Default: `evaluation/eval_results/generated_code`).
   - `--input-path` - Directory containing the JSON files describing the problems (Default: `evaluation/problem_json`).
   - `--prompt-dir` - Directory where prompt files are saved (Default: `evaluation/eval_results/prompt`).
   - `--temperature` - Controls the randomness of the generation (Default: 0).
    
 - ## **Evaluate generated code**

   Download `test_data.h5` at the path `evaluation/test_data.h5`.
   
   To run the script, go to the root of this repo and use the following command:
   
   ```bash
   python evaluation/scripts/test_generated_code.py
   ```
