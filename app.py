import os
from flask import Flask, request, jsonify, abort, send_file
from flask_cors import CORS
import logging
from dotenv import load_dotenv
# from pymongo.mongo_client import MongoClient
# import gridfs
from io import BytesIO
from argparse import ArgumentParser
import errno

# For handling line bot events
from line_handlers import handler
from linebot.v3.exceptions import InvalidSignatureError

import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Ensure necessary environment variables are set
required_env_vars = ['TYPHOON_API_KEY', 'LINE_CHANNEL_SECRET', 'LINE_CHANNEL_ACCESS_TOKEN']
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Please set the {var} environment variable in your .env file.")


# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Directory to save uploaded files
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# Create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise

chroma_settings = Settings()
chroma_client = chromadb.Client(chroma_settings)

# Ensure users collection exists
users_collection_name = "users"
existing_collections = chroma_client.list_collections()
if users_collection_name not in existing_collections:
    chroma_client.create_collection(users_collection_name)

users_collection = chroma_client.get_collection(users_collection_name)

shops_collection_name = "shops"
existing_collections = chroma_client.list_collections()
if shops_collection_name not in existing_collections:
    chroma_client.create_collection(shops_collection_name)

shops_collection = chroma_client.get_collection(shops_collection_name)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # Handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@app.route('/api/home', methods=['GET'])
def return_home():
    return jsonify({
        'message':"Test API Welcome",
    })

@app.route('/api/users', methods=['POST'])
def register_user():
    data = request.json

    # Handle form data
    name = data.get('name')
    surname = data.get('surname')
    age = data.get('age')
    weight = data.get('weight')
    height = data.get('height')

    ailment = data.get('ailment')
    allergies = data.get('allergies')

    favor = data.get('favor')
    disfavor = data.get('disfavor')
    avoid = data.get('avoid')

    line_user_id = data.get('lineUserId')

    if not name or not surname or not line_user_id:
        return jsonify({'error': 'Missing required form data'}), 400

    record = {
        'name': name,
        'surname': surname,
        'age': age,
        'weight': weight,
        'height': height,
        'ailment': ailment,
        'allergies': allergies,
        'favor': favor,
        'disfavor': disfavor,
        'avoid': avoid,
        'line_user_id': line_user_id
    }

    # embedding for users
    placeholder_embedding = [0.0] * 768

    try:
        users_collection.add(
            ids=[line_user_id],
            metadatas=[record],
            embeddings=[placeholder_embedding]
        )
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        app.logger.error(f"Error inserting user data: {e}")
        return jsonify({'error': 'Failed to register user', 'details': str(e)}), 500

@app.route('/api/shops', methods=['POST'])
def register_shop():
    data = request.json

    shop_name = data.get('name')
    phone_number = data.get('phoneNumber')
    social_media = data.get('socialMedia')
    location = data.get('location')

    # Handle file uploads
    image = request.files.get('image')
    file_menu = request.files.get('fileMenu')

    if not shop_name or not phone_number or not social_media or not location:
        return jsonify({'error': 'Missing required form data'}), 400

    # Prepare shop data
    shop_data = {
        'name': shop_name,
        'phone': phone_number,
        'social_media': social_media,
        'location': location,
       
    }
    print(shop_data)

    # Save uploaded files to a directory (e.g., 'uploads/')
    upload_dir = os.path.join(static_tmp_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    if image:
        image.save(os.path.join(upload_dir, image.filename))

    if file_menu:
        file_menu.save(os.path.join(upload_dir, file_menu.filename))

    # Embedding for shops
    placeholder_embedding = [0.0] * 768
    try:
        shops_collection.add(
            ids=[shop_name],
            metadatas=[shop_data],
            embeddings=[placeholder_embedding]
        )
        return jsonify({'message': 'Shop registered successfully'}), 201
    except Exception as e:
        app.logger.error(f"Error inserting shop data: {e}")
        return jsonify({'error': 'Failed to register shop', 'details': str(e)}), 500


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', type=int, default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    # Create tmp dir for download content
    make_static_tmp_dir()

    app.run(debug=options.debug, port=options.port)