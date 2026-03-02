import requests
import io
import json
import os
from pathlib import Path
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"


def _load_api_token():
    env_token = os.getenv("HF_API_TOKEN", "").strip()
    if env_token:
        return env_token

    key_file = Path(__file__).resolve().parent / "Key.json"
    if not key_file.exists():
        raise RuntimeError(
            "Hugging Face token not found. Create app/models/StableDiffusionAPI/Key.json "
            "or set HF_API_TOKEN."
        )

    with open(key_file) as file:
        content = json.load(file)

    if isinstance(content, list) and content:
        token = str(content[0]).strip()
    elif isinstance(content, dict) and "token" in content:
        token = str(content["token"]).strip()
    else:
        raise RuntimeError("Invalid Key.json format. Use [\"hf_xxx\"] or {\"token\": \"hf_xxx\"}.")

    if not token:
        raise RuntimeError("Hugging Face token is empty.")
    return token


def generate(text_prompt):
    token = _load_api_token()
    auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
    response = requests.post(
        API_URL,
        headers={"Authorization": auth_header},
        json={"inputs": text_prompt},
        timeout=90,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except ValueError:
            details = response.text[:300]
        raise RuntimeError(f"Hugging Face API error {response.status_code}: {details}")

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        try:
            details = response.json()
        except ValueError:
            details = response.text[:300]
        raise RuntimeError(f"Unexpected API response: {details}")

    return Image.open(io.BytesIO(response.content))


"""
text = input("Your sentence here:")
image = generate(text)
image.show()
"""
