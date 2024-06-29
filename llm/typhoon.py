import requests
import json

def get_typhoon_response(user_text, api_key, backstory, task):
    base_url = "https://api.opentyphoon.ai/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "typhoon-v1.5-instruct",
        "messages": [
          {
               "role": "system",
               "content": backstory
          },
          {
               "role": "user",
               "content": user_text,
          }
        ],
        "max_tokens": 500,
        "temperature": 0.6,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    result = ""
    for chunk in response.iter_lines():
        if chunk:
            result += chunk.decode('utf-8') + "\n"
    return result