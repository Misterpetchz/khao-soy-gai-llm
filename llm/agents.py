import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from llm.typhoon import get_typhoon_response
from dotenv import load_dotenv
import re
import sys
import string
from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize
from datetime import datetime
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.process import query_place_collection

# Load environment variables
load_dotenv()

# Set the Hugging Face token directly in the script (for testing purposes only)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Log in to Hugging Face using the token
# login(token=HUGGINGFACE_TOKEN)

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
        print("Preprocess_Data Done!!")
        return preprocessed_data

class RetrievalAgent:
    def __init__(self, csv_file):
        # self.data = pd.read_csv(csv_file)
        # self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        # self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", token=HUGGINGFACE_TOKEN)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-m3", token=HUGGINGFACE_TOKEN)

    # def encode(self, texts):
    #     inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
    #     return embeddings

    def retrieve_information(self, query, task_type):
        # # Encode the query
        # query_embedding = self.encode([query]).squeeze()

        # # Encode the documents
        # descriptions = self.data['description'].tolist()
        # doc_embeddings = self.encode(descriptions)

        # # Compute cosine similarity between the query and documents
        # similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)

        # # Get the top 5 most similar documents
        # top_k = similarities.topk(5)
        # top_docs = self.data.iloc[top_k.indices]
        if task_type != 'branding' or task_type !='unknown':     
            top_docs = query_place_collection(query, 5)
            recommendation_strings = []
            for idx, recommendation in enumerate(top_docs, 1):
                entry = (
                    f"{idx}. {recommendation['name']} at {recommendation['address']}\n"
                    f"   About Summary: {recommendation['editorial_summary']}\n"
                    f"   Types: {recommendation['types']}\n"
                    f"   Rating: {recommendation['rating']} (Total User Ratings: {recommendation['user_ratings_total']})\n"
                    f"   Price Level: {recommendation['price_level']}\n"
                    f"   Opening Hours: {recommendation['opening_hours']}\n"
                    f"   Dine-in: {recommendation['dine_in']}\n"
                    f"   Delivery: {recommendation['delivery']}\n"
                    f"   Takeout: {recommendation['takeout']}\n"
                    f"   Reviews: {recommendation['reviews'][:100]}{'...' if len(recommendation['reviews']) > 100 else ''}\n"
                    f"   Google Map Link: {recommendation['google_maps_link']}\n"
                )
                recommendation_strings.append(entry)

            # Print retrieved documents before re-ranking
            print("Retrieved documents before re-ranking:")
            # print(recommendation_strings)

            # Re-rank the top documents using a pretrained re-ranking model
            re_ranked_docs = self.re_rank(query, recommendation_strings)

            # Print re-ranked documents
            print("Re-ranked documents:")
            # print(re_ranked_docs)

            return re_ranked_docs.to_dict(orient='records')
        else:
            return ""

    def re_rank(self, query, docs):
        scores = []
        for doc in docs:
            inputs = self.reranker_tokenizer(query, doc, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
            score = outputs.logits.squeeze().tolist()
            scores.append(score)

        # Sort documents based on the scores
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return pd.DataFrame(ranked_docs, columns=['description'])

# class ChatHistoryAgent:
#     def __init__(self, log_file="chat_history.log"):
#         self.log_file = log_file

#     def log_interaction(self, user_input, system_response, userid):
#         with open(self.log_file, "a", encoding="utf-8") as f:
#             timestamp = datetime.now().isoformat()
#             f.write(f"{timestamp} - User[{userid}]: {user_input}\n")
#             f.write(f"{timestamp} - System: {system_response}\n\n")
#         print("ChatHistoryAgent Done!!")

class ChatHistoryAgent:
    def __init__(self, base_folder="user_database"):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)  # Ensure the base folder exists

    def ensure_header(self, log_file):
        # Check if the file is empty, if so write the header
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            self.write_header(log_file)

    def write_header(self, log_file):
        with open(log_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "userid", "user_chat", "system_ans"])

    def log_interaction(self, user_input, system_response, userid):
        timestamp = datetime.now().isoformat()
        log_file = os.path.join(self.base_folder, f"{userid}.csv")
        self.ensure_header(log_file)
        
        with open(log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, userid, user_input, system_response])
        print("ChatHistoryAgent Done!!")
        
    def get_all_interactions(self, userid):
        log_file = os.path.join(self.base_folder, f"{userid}.csv")
        if not os.path.exists(log_file):
            return "No interactions found for this user."

        interactions = []
        with open(log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_chat = row["user_chat"]
                system_ans = row["system_ans"]
                if isinstance(system_ans, str):
                    system_ans = json.loads(system_ans)
                    system_ans = system_ans['choices'][0]['message']['content']
                interactions.append(f"user:{user_chat}\nsystem:{system_ans}\n")

        return "\n --------------------------------------------------------------------".join(interactions)

class RestaurantAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        print("RestaurantAgent Done!!")
        return get_typhoon_response(combined_data, self.api_key, task="restaurant", backstory=self.backstory)

class FoodAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        print("FoodAgent Done!!")
        return get_typhoon_response(combined_data, self.api_key, task="food", backstory=self.backstory)

class BrandingAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info, backstory):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        print("BrandingAgent Done!!")
        return get_typhoon_response(combined_data, self.api_key, task="branding", backstory=self.backstory)

class PostprocessAgent:
    def postprocess(self, data):
        # Implement any postprocessing logic here (e.g., data formatting, summarization)
        print("Postprocess-Data Done!!")
        return data.strip()
