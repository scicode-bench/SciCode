from functools import partial
from openai import OpenAI
import anthropic
import google.generativeai as genai
import config
import re
import os
import litellm
from litellm.utils import validate_environment as litellm_validate_environment

from scicode import keys_cfg_path
from scicode.utils.log import get_logger

logger = get_logger("models")


def get_config():
    if not keys_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {keys_cfg_path}")
    return config.Config(str(keys_cfg_path))

def generate_litellm_response(prompt: str, *, model: str, **kwargs) -> str:
    """Call the litellm api to generate a response"""
    # litellm expects all keys as env variables
    config = get_config()
    for key, value in config.as_dict().items():
        if key in os.environ and os.environ[key] != value:
            logger.warning(f"Overwriting {key} from config with environment variable")
        else:
            os.environ[key] = value
    # Let's validate that we have everythong for this model
    env_validation = litellm_validate_environment(model)
    if not env_validation.get("keys_in_environment") or env_validation.get("missing_keys", []):
        msg = f"Environment validation for litellm failed for model {model}: {env_validation}"
        raise ValueError(msg)
    response = litellm.completion(
        model=model,
        messages = [
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return response.choices[0].message.content

def generate_openai_response(prompt: str, *, model="gpt-4-turbo-2024-04-09",
                             temperature: float = 0) -> str:
    """call the openai api to generate a response"""
    key: str = get_config()["OPENAI_KEY"]  # type: ignore
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def generate_anthropic_response(prompt, *, model="claude-3-opus-20240229",
                                max_tokens: int = 4096, temperature: float = 0) -> str:
    """call the anthropic api to generate a response"""
    key: str = get_config()["ANTHROPIC_KEY"]  # type: ignore
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return message.content[0].text


def generate_google_response(prompt: str, *, model: str = "gemini-pro",
                             temperature: float = 0) -> str:
    """call the api to generate a response"""
    key: str = get_config()["GOOGLE_KEY"]  # type: ignore
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name=model)
    response = model.generate_content(prompt,
                                      generation_config=genai.GenerationConfig(temperature=temperature),
                                      # safety_settings=[
                                      #     {
                                      #         "category": "HARM_CATEGORY_HARASSMENT",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_HATE_SPEECH",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                      #         "threshold": "BLOCK_NONE",
                                      #     },
                                      #     {
                                      #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                      #         "threshold": "BLOCK_NONE"
                                      #     }
                                      # ]
                                      )
    try:
        return response.text
    except ValueError:
        print(f'prompt:\n{prompt}')
        # If the response doesn't contain text, check if the prompt was blocked.
        print(f'prompt feedback:\n{response.prompt_feedback}')
        # Also check the finish reason to see if the response was blocked.
        print(f'finish reason:\n{response.candidates[0].finish_reason.name}')
        # If the finish reason was SAFETY, the safety ratings have more details.
        print(f'safety rating:\n{response.candidates[0].safety_ratings}')
        raise ValueError("Generate response failed.")


def get_model_function(model: str, **kwargs):
    """Return the appropriate function to generate a response based on the model"""
    if model.startswith("litellm/"):
        model = model.removeprefix("litellm/")
        fct = generate_litellm_response
    elif "gpt" in model:
        fct = generate_openai_response
    elif "claude" in model:
        fct = generate_anthropic_response
    elif "gemini" in model:
        fct = generate_google_response
    elif model == "dummy":
        fct = generate_dummy_response
    else:
        raise ValueError(f"Model {model} not supported")
    return partial(fct, model=model, **kwargs)


def generate_dummy_response(prompt: str, **kwargs) -> str:
    """Used for testing as a substitute for actual models"""
    return "Blah blah\n```python\nprint('Hello, World!')\n```\n"


def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script

