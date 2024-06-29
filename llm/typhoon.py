import requests
import json

def get_llm_prediction(user_text, api_key, backstory, model="typhoon-instruct",
                         max_tokens=300, temperature=0.6, top_p=1, top_k=50, 
                         repetition_penalty=1.15, stream=False):
    base_url = "https://api.opentyphoon.ai/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
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
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "stream": stream
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    predicted_task = result['choices'][0]['message']['content'].strip().lower()

    # Map the prediction to the appropriate agent
    if "restaurant" in predicted_task:
        return 'restaurant'
    elif "food" in predicted_task:
        return 'food'
    elif "branding" in predicted_task:
        return 'branding'
    else:
        return 'unknown'

def get_typhoon_response(user_text, api_key, backstory, task, 
                         max_tokens=500, temperature=0.7, top_p=1, top_k=50, 
                         repetition_penalty=1.15, stream=False):
    base_url = "https://api.opentyphoon.ai/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "typhoon-instruct",
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
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "stream": stream
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    result = ""
    for chunk in response.iter_lines():
        if chunk:
            result += chunk.decode('utf-8') + "\n"
    return result