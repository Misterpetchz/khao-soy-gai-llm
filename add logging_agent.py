import os
import requests
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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

class RetrievalAgent:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModel.from_pretrained("BAAI/bge-m3")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-m3")

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings

    def retrieve_information(self, query):
        # Encode the query
        query_embedding = self.encode([query]).squeeze()

        # Encode the documents
        descriptions = self.data['description'].tolist()
        doc_embeddings = self.encode(descriptions)

        # Compute cosine similarity between the query and documents
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)

        # Get the top 5 most similar documents
        top_k = similarities.topk(5)
        top_docs = self.data.iloc[top_k.indices]

        # Print retrieved documents before re-ranking
        print("Retrieved documents before re-ranking:")
        print(top_docs)

        # Re-rank the top documents using a pretrained re-ranking model
        re_ranked_docs = self.re_rank(query, top_docs['description'].tolist())

        # Print re-ranked documents
        print("Re-ranked documents:")
        print(re_ranked_docs)

        return re_ranked_docs.to_dict(orient='records')

    def re_rank(self, query, docs):
        scores = []
        for doc in docs:
            inputs = self.reranker_tokenizer(query, doc, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
            score = outputs.logits.squeeze().tolist()  # Correctly handle the logits output
            scores.append(score)

        # Sort documents based on the scores
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return pd.DataFrame(ranked_docs, columns=['description'])

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
        self.retrieval_agent = RetrievalAgent("data.csv")  # Pass the CSV file path to the RetrievalAgent
        self.restaurant_agent = RestaurantAgent(api_key)
        self.food_agent = FoodAgent(api_key)
        self.branding_agent = BrandingAgent(api_key)
        self.postprocess_agent = PostprocessAgent()
        self.chat_history_agent = ChatHistoryAgent()

    def predict_agent(self, data):
        # Use LLM to predict the appropriate agent
        prompt = (
            "คุณเป็นผู้เชี่ยวชาญในการจัดการคำขอ คุณสามารถช่วยในการจำแนกประเภทคำขอได้ "
            "กรุณาจำแนกประเภทคำขอที่ได้รับตามข้อมูลต่อไปนี้: "
            "1. restaurant (คำขอเกี่ยวกับร้านอาหาร)\n"
            "2. food (คำขอเกี่ยวกับอาหาร)\n"
            "3. branding (คำขอเกี่ยวกับการสร้างแบรนด์)\n"
            "\n"
            f"คำขอ: {data}\n"
            "กรุณาระบุประเภทคำขอ:"
        )
        response = get_llm_prediction(prompt, self.api_key)
        return response

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
                return {"error": "Unable to determine the appropriate agent"}

            # Perform retrieval to get relevant information
            retrieved_info = self.retrieval_agent.retrieve_information(preprocessed_data)

            if task_type == 'restaurant':
                response = self.restaurant_agent.handle_task(preprocessed_data, retrieved_info)
            elif task_type == 'food':
                response = self.food_agent.handle_task(preprocessed_data, retrieved_info)
            elif task_type == 'branding':
                response = self.branding_agent.handle_task(preprocessed_data, retrieved_info)
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

    def handle_task(self, data, retrieved_info):
        # Option to use top N documents
        top_n_docs = retrieved_info[:5]  # Use top 5 documents for example
        combined_data = data + "\n\n" + "ข้อมูลที่เกี่ยวข้อง: " + json.dumps(top_n_docs, ensure_ascii=False)
        prompt = self.construct_prompt(combined_data)
        return get_typhoon_response(prompt, self.api_key, task="restaurant")

    def construct_prompt(self, combined_data):
        # Advanced prompt engineering with few-shot learning and role-playing
        prompt = (
            "คุณเป็นนักวิจารณ์อาหารที่เชี่ยวชาญในด้านการแนะนำร้านอาหารในเชียงใหม่ "
            "คุณมีความรู้และประสบการณ์ในการแนะนำร้านอาหารที่มีชื่อเสียงและเป็นที่นิยม "
            "โดยเน้นร้านที่เสิร์ฟอาหารพื้นเมืองของเชียงใหม่ เช่น ข้าวซอย แกงฮังเล และน้ำพริกอ่อง "
            "และบรรยากาศดีสำหรับครอบครัว นี่คือข้อมูลที่เกี่ยวข้อง:\n\n"
            f"{combined_data}\n\n"
            "ตัวอย่าง:\n"
            "ร้านอาหาร: ข้าวซอยนิมมาน\n"
            "พิกัด: ซอยนิมมานเหมินท์ 17\n"
            "รีวิว: ร้านอาหารบรรยากาศดี มีเมนูข้าวซอยและแกงฮังเลที่อร่อยมาก เหมาะสำหรับครอบครัว\n"
            "ราคา: ราคาเริ่มต้นที่ 100 บาท\n\n"
            "กรุณาแนะนำร้านอาหารท้องถิ่นในเชียงใหม่พร้อมบอกชื่อร้าน พิกัดสถานที่ตั้ง รีวิวของร้านอาหาร และราคา กรุณาตอบเป็นภาษาไทย"
        )
        return prompt

class FoodAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "ข้อมูลที่เกี่ยวข้อง: " + json.dumps(retrieved_info, ensure_ascii=False)
        prompt = self.construct_prompt(combined_data)
        return get_typhoon_response(prompt, self.api_key, task="food")

    def construct_prompt(self, combined_data):
        # Advanced prompt engineering with few-shot learning and role-playing
        prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านอาหารพื้นเมืองของเชียงใหม่ "
            "คุณมีความรู้เกี่ยวกับสูตรอาหารและวิธีการทำอาหารที่เป็นที่นิยม เช่น ข้าวซอย แกงฮังเล และน้ำพริกอ่อง "
            "นี่คือข้อมูลที่เกี่ยวข้อง:\n\n"
            f"{combined_data}\n\n"
            "ตัวอย่าง:\n"
            "เมนู: ข้าวซอย\n"
            "วิธีทำ: ผสมเครื่องแกงข้าวซอยลงในหม้อน้ำเดือด ใส่ไก่และกะทิ เคี่ยวจนไก่สุกเสิร์ฟพร้อมเส้นหมี่และเครื่องเคียง\n"
            "เคล็ดลับ: ใส่กะทิทีละน้อยเพื่อให้รสชาติกลมกล่อม\n\n"
            "กรุณาให้คำแนะนำเกี่ยวกับอาหารพื้นเมืองของเชียงใหม่พร้อมบอกรายละเอียดของแต่ละเมนูและวิธีการทำ กรุณาตอบเป็นภาษาไทย"
        )
        return prompt

class BrandingAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "ข้อมูลที่เกี่ยวข้อง: " + json.dumps(retrieved_info, ensure_ascii=False)
        prompt = self.construct_prompt(combined_data)
        return get_typhoon_response(prompt, self.api_key, task="branding")

    def construct_prompt(self, combined_data):
        # Advanced prompt engineering with few-shot learning and role-playing
        prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการสร้างแบรนด์และการตลาดสำหรับร้านอาหารในเชียงใหม่ "
            "คุณมีความรู้เกี่ยวกับกลยุทธ์การตลาดและการโปรโมทร้านอาหารผ่านช่องทางต่างๆ "
            "นี่คือข้อมูลที่เกี่ยวข้อง:\n\n"
            f"{combined_data}\n\n"
            "ตัวอย่าง:\n"
            "กลยุทธ์: การสร้างแบรนด์ร้านอาหาร\n"
            "รายละเอียด: สร้างเอกลักษณ์ที่ชัดเจนและแตกต่างให้กับร้านอาหาร ใช้สื่อโซเชียลมีเดียในการโปรโมทและสร้างการรับรู้\n"
            "เคล็ดลับ: เน้นการใช้ภาพถ่ายที่ดึงดูดความสนใจและการสื่อสารที่ตรงประเด็น\n\n"
            "กรุณาให้คำแนะนำเกี่ยวกับการสร้างแบรนด์และการตลาดสำหรับร้านอาหารในเชียงใหม่ กรุณาตอบเป็นภาษาไทย"
        )
        return prompt

class PostprocessAgent:
    def postprocess(self, data):
        # Implement any postprocessing logic here (e.g., data formatting, summarization)
        return data.strip()

def get_llm_prediction(prompt, api_key):
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
        "max_tokens": 300,
        "temperature": 0.6,
        "top_p": 1
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

def get_typhoon_response(prompt, api_key, task):
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

    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    result = ""
    for chunk in response.iter_lines():
        if chunk:
            result += chunk.decode('utf-8') + "\n"
    return result

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
        print(json.dumps(response, indent=4, ensure_ascii=False))

# Start the QA loop
qa_loop()
