# Gemini Report: SciCode

This document provides an overview of the SciCode project, focusing on the supported models and how to run them.

## Supported Models

The SciCode project supports a variety of Large Language Models through different providers. The core logic for model integration is located in `src/scicode/gen/models.py`.

The following model providers are directly supported:

*   **OpenAI:** Models with "gpt" in their name (e.g., `gpt-4o`, `gpt-4-turbo-2024-04-09`).
*   **Anthropic:** Models with "claude" in their name (e.g., `claude-3-opus-20240229`).
*   **Google:** Models with "gemini" in their name (e.g., `gemini-pro`).
*   **LiteLLM:** A wide range of models can be used through LiteLLM by prefixing the model name with `litellm/`. For a list of models that have been evaluated with this benchmark, please refer to the leaderboard in the `README.md`.
*   **Dummy:** A `dummy` model is available for testing purposes.

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
    Create a configuration file at `~/.config/scicode/keys.cfg` and add your API keys. See the "Configuration" section below for more details.

5.  **Run the evaluation:**
    You can run the evaluation using the `inspect` command. The model name should be in the format `<provider>/<model_name>`.

    ```bash
    inspect eval eval/inspect_ai/scicode.py --model openai/gpt-4o --temperature 0
    ```

### Deprecated Method

A deprecated two-step process is also available:

1.  **Generate code:**
    ```bash
    python eval/scripts/gencode.py --model <model_name>
    ```
2.  **Test the generated code:**
    ```bash
    python eval/scripts/test_generated_code.py
    ```

## Configuration

The API keys for the different model providers are managed in a configuration file located at `~/.config/scicode/keys.cfg`.

Create the directory and the file if they don't exist:
```bash
mkdir -p ~/.config/scicode
touch ~/.config/scicode/keys.cfg
```

Then, add your API keys to the `keys.cfg` file in the following format:

```
OPENAI_KEY = "your-openai-api-key"
ANTHROPIC_KEY = "your-anthropic-api-key"
GOOGLE_KEY = "your-google-api-key"
# Add other keys for LiteLLM as needed
```
