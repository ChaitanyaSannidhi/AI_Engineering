
import os
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import whoami

# 1. Load environment variables
load_dotenv(override=True)

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found in .env file!")

# 2. Verify Hugging Face authentication
user = whoami(token=token)
print("Authenticated account:", user["name"])

# 3. Initialize OpenAI-compatible Hugging Face client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=token
)

# 4. Select model with automatic provider routing
MODEL = "Qwen/Qwen2.5-72B-Instruct:fastest"

# 5. Test chat completion
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain synthetic data generation in simple terms."
        }
    ],
    max_tokens=150
)

# 6. Print result
print("\nModel response:")
print(response.choices[0].message.content)
