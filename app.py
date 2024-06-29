import os
from flask import Flask, request, jsonify, abort, send_file
from flask_cors import CORS
import logging
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import gridfs
from io import BytesIO
from argparse import ArgumentParser
import errno

# For handling line bot events
from line_handlers import handler
from linebot.v3.exceptions import InvalidSignatureError

# Load environment variables
load_dotenv()

# Ensure necessary environment variables are set
required_env_vars = ['MONGO_URI', 'TYPHOON_API_KEY', 'LINE_CHANNEL_SECRET', 'LINE_CHANNEL_ACCESS_TOKEN']
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Please set the {var} environment variable in your .env file.")

# Connect to MongoDB with `pymongo[srv]`
# mongo_uri = os.getenv('MONGO_URI')
# client = MongoClient(mongo_uri)
# db = client['maemanee']  # Use your MongoDB database name
# grid_fs = gridfs.GridFS(db)
# collection = db['registrations']

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

# @app.route('/api/register', methods=['POST'])
# def register_user():
#     # Handle form data
#     shop_name = request.form.get('shopName')
#     phone = request.form.get('phone')
#     social_media = request.form.get('socialMedia')
#     location = request.form.get('location')

#     # Handle file upload
#     menu_file = request.files.get('menuFile')
#     file_id = None

#     if menu_file:
#         file_id = grid_fs.put(menu_file, filename=menu_file.filename, contentType=menu_file.content_type)

#     user_id = request.form.get('userId')
#     display_name = request.form.get('displayName')
#     picture_url = request.form.get('pictureUrl')
#     status_message = request.form.get('statusMessage')

#     # Create record to save in MongoDB
#     record = {
#         'shopName': shop_name,
#         'phone': phone,
#         'socialMedia': social_media,
#         'location': location,
#         'userId': user_id,
#         'displayName': display_name,
#         'pictureUrl': picture_url,
#         'statusMessage': status_message,
#         'fileId': file_id
#     }

#     # Insert record into MongoDB
#     result = collection.insert_one(record)

#     # Return the response
#     return jsonify({
#         'status': 'success',
#         'data': {
#             'shopName': shop_name,
#             'phone': phone,
#             'socialMedia': social_media,
#             'location': location,
#             'userId': user_id,
#             'displayName': display_name,
#             'pictureUrl': picture_url,
#             'statusMessage': status_message,
#             'fileId': str(file_id),
#             'id': str(result.inserted_id)
#         }
#     }), 201

# @app.route('/file/<file_id>', methods=['GET'])
# def get_file(file_id):
#     try:
#         grid_out = grid_fs.get_last_version(file_id)
#         return send_file(BytesIO(grid_out.read()), attachment_filename=grid_out.filename, mimetype=grid_out.content_type)
#     except gridfs.NoFile:
#         abort(404)

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