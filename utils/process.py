import logging
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
# import cohere
import requests
import os
import time
from dotenv import load_dotenv
from chromadb.utils import embedding_functions


# Configure logging
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setFormatter(formatter)

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Updated ChromaDB client for local usage
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# # Custom embedding function
# class CustomCohereEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, api_key, model_name):
#         self.client = cohere.Client(api_key)
#         self.model_name = model_name
    
#     def __call__(self, input: Documents) -> Embeddings:
#         embeddings = self.client.embed(
#             texts=input,
#             model=self.model_name,
#             input_type="search_query"  
#         ).embeddings

#         # Ensure embeddings are lists
#         return [list(embedding) if isinstance(embedding, tuple) else embedding for embedding in embeddings]

# # Instantiate the custom embedding function
# custom_cohere_ef = CustomCohereEmbeddingFunction(
#     api_key=COHERE_API_KEY,
#     model_name="embed-multilingual-v3.0"
# )
model_name = 'BAAI/bge-m3'
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


# Create or get the place collection
place_collection = client.get_or_create_collection(
    name = "place_collection",
    embedding_function = sentence_transformer_ef,
    metadata = {"hnsw:space": "cosine"}
)

def search_places(api_key, location, keywords, limit=1000):
    """Search for places to eat in a specific location using the Google Maps API."""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    results = []
    unique_place_ids = set()

    for keyword in keywords:
        query = f"{keyword} in {location}"
        params = {
            "key": api_key,
            "query": query,
            "maxResults": limit,
            "language": "th",
            "region": "th",
        }

        while True:
            response = requests.get(url, params=params)
            logging.info(f"Response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                keyword_results = data.get("results", [])[:limit]

                for place in keyword_results:
                    place_id = place.get("place_id")
                    if place_id and place_id not in unique_place_ids:
                        results.append(place)
                        unique_place_ids.add(place_id)
                        logging.info(f"Added place: {place['name']} (Keyword: {keyword})")

                next_page_token = data.get("next_page_token")
                if next_page_token:
                    params["pagetoken"] = next_page_token
                    time.sleep(2)
                else:
                    break
            else:
                logging.error(f"Request failed with status code {response.status_code}")
                break
    return results

def get_place_details(api_key, place_id):
    """Get details for a specific place using the Google Maps API."""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "key": api_key,
        "place_id": place_id,
        "language": "th",
        "region": "th",
        "fields": "name,formatted_address,rating,user_ratings_total,types,price_level,website,reviews,"
                  "opening_hours,url,vicinity,place_id,editorial_summary,dine_in,delivery,takeout,photos"
    }
    response = requests.get(url, params=params)
    results = response.json()
    return results.get("result", {})

def get_photo_url(api_key, photo_reference):
    """Generate photo URL from photo reference using the Google Maps API."""
    return f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"

def get_google_maps_link(place_id):
    """Generate a Google Maps link for a place using the place_id."""
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}"

def process_place_data(place_data, api_key):
    """Process place data to create documents, metadata, and ids."""
    documents = []
    metadata = []
    ids = []

    for place in place_data:
        opening_hours_str = ', '.join(place.get('opening_hours', {}).get('weekday_text', []))
        reviews_list = place.get('reviews', [])
        reviews_str = '; '.join([f"({review['rating']} stars): {review['text']}" for review in reviews_list]).replace('\n', ' ').replace('\r', ' ')
        editorial_summary = place.get('editorial_summary', {}).get('overview', 'N/A')
        photo_references = place.get('photos', [])
        photo_urls = [get_photo_url(api_key, photo['photo_reference']) for photo in photo_references]
        photo_urls_str = '; '.join(photo_urls)

        document_text = (
            f"Address: {place['formatted_address']}\nAbout: {editorial_summary}\nTypes: {', '.join(place['types'])}\n"
            f"Number of ratings: {place.get('user_ratings_total', 'N/A')}\nOpening hours: {opening_hours_str}\n"
            f"Price Level: {place.get('price_level', 'N/A')}\n"
            f"Dine In: {place.get('dine_in', 'N/A')}\nDelivery: {place.get('delivery', 'N/A')}\nTakeout: {place.get('takeout', 'N/A')}"
        )

        documents.append(document_text)
        logging.info(f"\nDocument text: {document_text}")

        place_metadata = {
            'name': place['name'],
            'address': place['formatted_address'],
            'types': ', '.join(place['types']),
            'rating': place.get('rating') or 'N/A',
            'user_ratings_total': place.get('user_ratings_total') or 'N/A',
            'opening_hours': opening_hours_str if opening_hours_str else 'N/A',
            'reviews': reviews_str if reviews_str else 'N/A',
            'editorial_summary': editorial_summary,
            'price_level': place.get('price_level') or 'N/A',
            'dine_in': place.get('dine_in', 'N/A'),
            'delivery': place.get('delivery', 'N/A'),
            'takeout': place.get('takeout', 'N/A'),
            'photo_urls': photo_urls_str,
            'google_maps_link': get_google_maps_link(place['place_id'])
        }
        metadata.append(place_metadata)
        logging.info(f"Metadata: {len(place_metadata)}\n")
        ids.append(place['place_id'])

    return documents, metadata, ids

def fetch_and_store_place_collection(location):
    """Fetch and store place data."""
    cities = ["chiang mai"]
    keywords = ["restaurant", "food", "cafe", "breakfast", "brunch", "dinner", "lunch", "dessert"]

    for location in cities:
        logging.info(f"Searching for places to eat in {location}...")
        places = search_places(GOOGLE_MAPS_API_KEY, location=location, keywords=keywords)
        logging.info(f"Found {len(places)} places to eat")

        place_data = []
        existing_ids = place_collection.get()['ids']

        for place in places:
            place_id = place["place_id"]
            if place_id not in existing_ids:
                logging.info(f"Processing place: {place['name']}")
                details = get_place_details(GOOGLE_MAPS_API_KEY, place_id)
                place_data.append(details)
            else:
                logging.info(f"Skipping existing place: {place['name']}")

        logging.info(f"Processed {len(place_data)} places to eat")
        place_documents, place_metadata, place_ids = process_place_data(place_data, GOOGLE_MAPS_API_KEY)
        places_to_add = [i for i in range(len(place_ids)) if place_ids[i] not in existing_ids]

        filtered_place_documents = [place_documents[i] for i in places_to_add]
        filtered_place_metadata = [place_metadata[i] for i in places_to_add]
        filtered_place_ids = [place_ids[i] for i in places_to_add]

        logging.info(f"Number of places in the collection before adding: {place_collection.count()}")
        logging.info(f"Number of places to be added: {len(filtered_place_documents)}")

        batch_size = 166  # Maximum batch size
        for start_index in range(0, len(filtered_place_documents), batch_size):
            end_index = start_index + batch_size
            batch_documents = filtered_place_documents[start_index:end_index]
            batch_metadata = filtered_place_metadata[start_index:end_index]
            batch_ids = filtered_place_ids[start_index:end_index]

            if batch_documents and batch_metadata and batch_ids:
                place_collection.add(documents=batch_documents, metadatas=batch_metadata, ids=batch_ids)
            else:
                logging.warning("No new places to add in this batch.")

        logging.info(f"Number of places in the collection after adding: {place_collection.count()}")

def query_place_collection(query, num_results=3):
    """Query the place collection in ChromaDB."""
    metadata_list = []
    results = place_collection.query(query_texts=[query], n_results=num_results)
    results_metadata = results['metadatas'][0]
    for result in results_metadata:
        metadata = {
            'name': result.get('name'),
            'address': result.get('address'),
            'types': result.get('types'),
            'rating': result.get('rating'),
            'user_ratings_total': result.get('user_ratings_total'),
            'price_level': result.get('price_level'),
            'opening_hours': result.get('opening_hours'),
            'reviews': result.get('reviews'),
            'editorial_summary': result.get('editorial_summary'),
            'dine_in': result.get('dine_in'),
            'delivery': result.get('delivery'),
            'takeout': result.get('takeout'),
            'photo_urls': result.get('photo_urls', []),
            'google_maps_link': result.get('google_maps_link')
        }
        metadata_list.append(metadata)

    return metadata_list