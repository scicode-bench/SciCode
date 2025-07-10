from openai import OpenAI
import config
import re
import os

from scicode import keys_cfg_path
from scicode.utils.log import get_logger

logger = get_logger("models")


def get_config():
    if not keys_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {keys_cfg_path}")
    return config.Config(str(keys_cfg_path))


def get_model_response(prompt: str, *, model: str) -> str:
    """Call the OpenRouter api to generate a response, or use a dummy for testing."""
    if model == "dummy":
        return generate_dummy_response(prompt)

    key: str = get_config()["OPENROUTER_KEY"]  # type: ignore
    if not key:
        raise ValueError(
            "OPENROUTER_KEY not found in config file. Please add it to your keys.cfg."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def generate_dummy_response(prompt: str) -> str:
    """Used for testing as a substitute for actual models"""
    return "Blah blah\n```python\nprint('Hello, World!')\n```\n"


def extract_python_script(response: str):
    # We will extract the python script from the response
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE
    )
    return python_script

