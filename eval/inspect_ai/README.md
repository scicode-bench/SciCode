## **SciCode Evaluation using `inspect_ai`**

### 1. Set Up Your API Keys

Users can follow [`inspect_ai`'s official documentation](https://inspect.ai-safety-institute.org.uk/#getting-started) to setup correpsonding API keys depending on the types of models they would like to evaluate.

### 2. Setup Command Line Arguments if Needed

In most cases, after users setting up the key, they can directly start the SciCode evaluation via the following command.

```bash
inspect eval scicode.py --model <your_model> --temperature 0
```

However, there are some additional command line arguments that could be useful as well.

- `--max_connections`: Maximum amount of API connections to the evaluated model.
- `--limit`: Limit of the number of samples to evaluate in the SciCode dataset.
- `-T input_path=<another_input_json_file>`: This is useful when user wants to change to another json dataset (e.g., the dev set).
- `-T output_dir=<your_output_dir>`: This changes the default output directory (`./tmp`).
- `-T with_background=True/False`: Whether to include problem background.
- `-T mode=normal/gold/dummy`: This provides two additional modes for sanity checks.
    - `normal` mode is the standard mode to evaluate a model
    - `gold` mode can only be used on the dev set which loads the gold answer
    - `dummy` mode does not call any real LLMs and generates some dummy outputs

For example, user can run five sames on the dev set with background as

```bash
inspect eval scicode.py \
    --model openai/gpt-4o \
    --temperature 0 \
    --limit 5 \
    -T input_path=../data/problems_dev.jsonl \
    -T output_dir=./tmp/dev \
    -T with_background=True \
    -T mode=gold
```

For more information regarding `inspect_ai`, we refer users to its [official documentation](https://inspect.ai-safety-institute.org.uk/).