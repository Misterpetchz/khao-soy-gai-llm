import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from typhoon import get_typhoon_response
from huggingface_hub import login
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.process import query_place_collection

# Load environment variables
load_dotenv()

# Set the Hugging Face token directly in the script (for testing purposes only)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Log in to Hugging Face using the token
# login(token=HUGGINGFACE_TOKEN)

class PreprocessAgent:
    def preprocess(self, data):
        # Implement any preprocessing logic here (e.g., data cleaning, formatting)
        return data.strip()

class RetrievalAgent:
    def __init__(self, csv_file):
        # self.data = pd.read_csv(csv_file)
        # self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        # self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", token=HUGGINGFACE_TOKEN)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-large-en-v1.5", token=HUGGINGFACE_TOKEN)

    # def encode(self, texts):
    #     inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
    #     return embeddings

    def retrieve_information(self, query):
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
        print(top_docs)

        # Re-rank the top documents using a pretrained re-ranking model
        re_ranked_docs = self.re_rank(query, recommendation_strings)

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
            score = outputs.logits.item()
            scores.append(score)

        # Sort documents based on the scores
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return pd.DataFrame(ranked_docs, columns=['description'])

class RestaurantAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        return get_typhoon_response(combined_data, self.api_key, task="restaurant", backstory=self.backstory)

class FoodAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        return get_typhoon_response(combined_data, self.api_key, task="food", backstory=self.backstory)

class BrandingAgent:
    def __init__(self, api_key, backstory):
        self.api_key = api_key
        self.backstory = backstory

    def handle_task(self, data, retrieved_info, backstory):
        combined_data = data + "\n\n" + "Relevant information: " + json.dumps(retrieved_info, ensure_ascii=False)
        return get_typhoon_response(combined_data, self.api_key, task="branding", backstory=self.backstory)

class PostprocessAgent:
    def postprocess(self, data):
        # Implement any postprocessing logic here (e.g., data formatting, summarization)
        return data.strip()