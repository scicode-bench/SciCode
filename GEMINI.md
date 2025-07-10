# Gemini Report: SciCode

This document provides an overview of the SciCode project, focusing on the supported models and how to run them.

## Supported Models

The SciCode project uses [OpenRouter](https://openrouter.ai/) to support a wide variety of Large Language Models. The core logic for model integration is located in `src/scicode/gen/models.py`.

Any model available through the OpenRouter API can be used by providing its model identifier (e.g., `openai/gpt-4o`, `anthropic/claude-3-opus`). For a list of available models, please refer to the [OpenRouter documentation](https://openrouter.ai/docs#models).

A `dummy` model is also available for testing purposes.

## Running the Models

The recommended way to evaluate a new model is by using `inspect_ai`, as suggested in the `README.md`.

### Using `inspect_ai` (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:scicode-bench/SciCode.git
    cd SciCode
    ```

2.  **Install the package:**
    ```bash
    pip install -e .
    ```

3.  **Download test data:**
    Download the numeric test results from the link provided in the `README.md` and save them as `./eval/data/test_data.h5`.

4.  **Configure API Keys:**
    Set your OpenRouter API key as an environment variable. See the "Configuration" section below for more details.

5.  **Run the evaluation:**
    You can run the evaluation using the `inspect` command. The model name should be a valid OpenRouter model identifier.

    ```bash
    inspect eval eval/inspect_ai/scicode.py --model openai/gpt-4o --temperature 0
    ```

### Deprecated Method

A deprecated two-step process is also available:

1.  **Generate code:**
    ```bash
    python eval/scripts/gencode.py --model <openrouter_model_name>
    ```
2.  **Test the generated code:**
    ```bash
    python eval/scripts/test_generated_code.py
    ```

## Configuration

To use the models, you need to set your OpenRouter API key as an environment variable.

```bash
export OPENROUTER_KEY="your-openrouter-api-key"
```

You can add this line to your shell's startup file (e.g., `~/.bashrc` or `~/.zshrc`) to make it permanent.