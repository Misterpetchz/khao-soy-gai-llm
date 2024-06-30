import os
import requests
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from huggingface_hub import login
import re
import string
from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize
from datetime import datetime

# Set the Hugging Face token directly in the script (for testing purposes only)
huggingface_token = "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"

# Log in to Hugging Face using the token
login(token=huggingface_token)

# Set the API key directly in the script (for testing purposes only)
os.environ["OPENTYPHOON_API_KEY"] = "sk-pbtHFB2O8idAeyrz2PqRqZk5c8LpX7CRYCzQqcswUZ9cofPk"

# Ensure the environment variable is set correctly
api_key = os.environ.get("OPENTYPHOON_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENTYPHOON_API_KEY environment variable.")

class PreprocessAgent:
    def __init__(self):
        self.stop_words = set(thai_stopwords())

    def preprocess(self, data):
        # Normalize text
        data = normalize(data)
        # Remove special characters and punctuation
        data = re.sub(f"[{re.escape(string.punctuation)}]", '', data)
        # Tokenize
        tokens = thai_word_tokenize(data)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Join tokens back to string
        preprocessed_data = ' '.join(tokens)
        return preprocessed_data

class ChatHistoryAgent:
    def __init__(self, log_file="chat_history.log"):
        self.log_file = log_file

    def log_interaction(self, user_input, system_response):
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} - User: {user_input}\n")
            f.write(f"{timestamp} - System: {system_response}\n\n")

class ManagerAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.preprocess_agent = PreprocessAgent()
        self.restaurant_agent = RestaurantAgent(api_key)
        self.food_agent = FoodAgent(api_key)
        self.branding_agent = BrandingAgent(api_key)
        self.postprocess_agent = PostprocessAgent()
        self.chat_history_agent = ChatHistoryAgent()

    def predict_agent(self, data):
        # Simple heuristic-based method to predict the agent based on keywords
        data_lower = data.lower()
        if "อาหาร" in data_lower or "ข้าวซอย" in data_lower or "แกงฮังเล" in data_lower:
            return 'food'
        elif "ร้านอาหาร" in data_lower or "บรรยากาศ" in data_lower:
            return 'restaurant'
        elif "แบรนด์" in data_lower or "การตลาด" in data_lower:
            return 'branding'
        else:
            return 'unknown'

    def handle_request(self, request):
        try:
            data = request.get("data")

            if not data:
                return {"error": "Missing data"}

            # Preprocess the data
            preprocessed_data = self.preprocess_agent.preprocess(data)

            # Predict the appropriate agent based on the preprocessed data
            task_type = self.predict_agent(preprocessed_data)

            if task_type == 'unknown':
                response = get_llm_response(data, self.api_key)
                task_type = self.predict_agent(response)

            if task_type == 'unknown':
                return {"error": "Unable to determine the appropriate agent"}

            # Handle the request with the appropriate agent
            if task_type == 'restaurant':
                response = self.restaurant_agent.handle_task(preprocessed_data)
            elif task_type == 'food':
                response = self.food_agent.handle_task(preprocessed_data)
            elif task_type == 'branding':
                response = self.branding_agent.handle_task(preprocessed_data)
            else:
                return {"error": "Invalid task type"}

            # Postprocess the response
            final_response = self.postprocess_agent.postprocess(response)

            # Log the interaction
            self.chat_history_agent.log_interaction(data, final_response)

            return final_response

        except Exception as e:
            return {"error": str(e)}

class RestaurantAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def handle_task(self, data):
        system_prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการแนะนำร้านอาหารและการเขียนคำอธิบายสำหรับเนื้อหาของร้านอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความชอบของผู้ใช้และแนะนำร้านอาหารที่เหมาะสมที่สุด "
            "เวลาพูดเน้นเรื่องร้านอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำอธิบายที่น่าสนใจสำหรับเนื้อหาของร้านอาหารเพื่อดึงดูดลูกค้า "
            "คำอธิบายควรรวมถึงรายละเอียดเกี่ยวกับอาหารแต่ละจาน เช่น ส่วนผสม รสชาติ และบรรยากาศของร้าน "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการเลือกอาหารที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำร้านอาหารสำหรับครอบครัว บรรยากาศดี ราคาไม่แพง เป็นต้น"
        )
        combined_data = system_prompt + "\n\n" + "คำขอ: " + data
        return get_typhoon_response(combined_data, self.api_key, task="restaurant")

class FoodAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def handle_task(self, data):
        system_prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการแนะนำอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความชอบของผู้ใช้และแนะนำอาหารที่เหมาะสมที่สุด "
            "เวลาพูดเน้นเรื่องอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำอธิบายที่น่าสนใจสำหรับอาหารเพื่อดึงดูดลูกค้า "
            "คำอธิบายควรรวมถึงรายละเอียดเกี่ยวกับอาหารแต่ละจาน เช่น ส่วนผสม รสชาติ และประสบการณ์การรับประทาน "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการเลือกอาหารที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำอาหารท้องถิ่น อาหารที่เป็นเอกลักษณ์ และอาหารตามฤดูกาล เป็นต้น"
        )
        combined_data = system_prompt + "\n\n" + "คำขอ: " + data
        return get_typhoon_response(combined_data, self.api_key, task="food")

class BrandingAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def handle_task(self, data):
        system_prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการสร้างแบรนด์สำหรับร้านอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความต้องการและเป้าหมายของผู้ใช้ในการสร้างแบรนด์ "
            "เวลาพูดเน้นเรื่องอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำแนะนำที่น่าสนใจสำหรับการสร้างแบรนด์ที่โดดเด่น "
            "คำแนะนำควรรวมถึงรายละเอียดเกี่ยวกับกลยุทธ์การตลาด การสร้างความเป็นเอกลักษณ์ของแบรนด์ และวิธีการดึงดูดลูกค้า "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการสร้างแบรนด์ที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำการใช้สื่อสังคมออนไลน์ การจัดกิจกรรม และการออกแบบบรรจุภัณฑ์ เป็นต้น"
        )
        combined_data = system_prompt + "\n\n" + "คำขอ: " + data
        return get_typhoon_response(combined_data, self.api_key, task="branding")

class PostprocessAgent:
    def postprocess(self, data):
        # Implement any postprocessing logic here (e.g., data formatting, summarization)
        return data.strip()

def get_llm_response(prompt, api_key):
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
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 500,
        "temperature": 0.6,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result['choices'][0]['message']['content'].strip()

def qa_loop():
    manager = ManagerAgent(api_key)

    print("Welcome to the QA system. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting the QA system. Goodbye!")
            break

        request = {"data": user_input}
        response = manager.handle_request(request)

        print("System: ")
        print(response)

# Start the QA loop
qa_loop()
