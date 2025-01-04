import os, json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from functools import wraps
import dotenv
from modal import Image, App, Secret, asgi_app
from openai import AsyncOpenAI, RateLimitError, OpenAIError

# Unstructured API imports
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig

app = Flask(__name__)

# Create a Modal App and Image with the required dependencies
modal_app = App("teli-context-message-generator")
image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

# Get environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")
# print(pinecone_api_key, pinecone_index_name, openai_api_key)

# Unstructured API environment variables
unstructured_api_key = os.environ.get("UNSTRUCTURED_API_KEY")
unstructured_api_url = os.environ.get("UNSTRUCTURED_API_URL")
local_file_input_dir = os.environ.get("LOCAL_FILE_INPUT_DIR")
local_file_output_dir = os.environ.get("LOCAL_FILE_OUTPUT_DIR")
# print(unstructured_api_key, unstructured_api_url, local_file_input_dir, local_file_output_dir)

# Set the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, local_file_input_dir)
OUTPUT_DIR = os.path.join(BASE_DIR, local_file_output_dir)
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per chunk

# Ensure input and output directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize a Pinecone client and OpenAI client with the API keys
aclient = AsyncOpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# Create a Pinecone index if it doesn't exist
if pc.list_indexes().indexes[0].name != pinecone_index_name:
    pc.create_index(
        name=pinecone_index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Index {pinecone_index_name} created successfully!")
else:
    print(f"Index {pinecone_index_name} already exists.")

# Wait for the index to be ready
while not pc.describe_index(pinecone_index_name).status['ready']:
    time.sleep(1)

def get_api_key():
    return os.environ.get("API_KEY")

def require_api_key(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') and request.headers.get('X-API-Key') == get_api_key():
            return await f(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized"}), 401

    return decorated_function

def check_file_availability(file, context, endpoint):
    arr = []
    counter = 1

    if file is not None :
        with open(f"{os.path.join(OUTPUT_DIR, endpoint)}.json", "r") as f:
            content = json.load(f)
            for obj in content:
                arr.append({
                    "id": f"vec{counter}",
                    "text": obj['text']
                })
                counter += 1

    if context is not None:
        for text in context:
            arr.append({
                "id": f"vec{counter}",
                "text": text
            })
            counter += 1

    return arr

@app.route('/ingest-teli-data', methods=['POST'])
# @require_api_key
def ingest_teli_data():
    try:
        if "unique_id" not in request.form and ("file" not in request.files or "context" not in request.files):
            return jsonify({"error": "Missing required fields"}), 400

        unique_id = request.form.get("unique_id")
        context_str = request.form.get("context") if "context" in request.form else None
        file = request.files['file'] if 'file' in request.files else None

        # Parse context string
        context_arr = json.loads(context_str) if context_str is not None else None

        filename = None
        file_extension = None
        input_endpoint = None

        if file is not None:
            # Secure the filename
            filename = secure_filename(file.filename)
            file_extension = os.path.splitext(filename)[1].lower()
            input_endpoint = f"{unique_id}-{filename}"

            if file_extension not in [".pdf"]:
                return jsonify({"error": "Unsupported file type"}), 400

            # Save the uploaded file in the input directory
            input_file_path = os.path.join(INPUT_DIR, input_endpoint)
            file.save(input_file_path)

            # Use the Unstructured Pipeline to process the file
            Pipeline.from_configs(
                context=ProcessorConfig(),
                indexer_config=LocalIndexerConfig(input_path=INPUT_DIR),
                downloader_config=LocalDownloaderConfig(),
                source_connection_config=LocalConnectionConfig(),
                partitioner_config=PartitionerConfig(
                    partition_by_api=True,
                    api_key=unstructured_api_key,
                    partition_endpoint=unstructured_api_url,
                    strategy="hi_res",
                    additional_partition_args={
                        "split_pdf_page": True,
                        "split_pdf_allow_failed": True,
                        "split_pdf_concurrency_level": 15
                    }
                ),
                uploader_config=LocalUploaderConfig(output_dir=OUTPUT_DIR)
            ).run()

        # Read processed output files
        context = check_file_availability(file, context_arr, input_endpoint)

        # Convert the text into numerical vectors for Pinecone
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[c['text'] for c in context],
            parameters={"input_type": "passage", "truncate": "END"}
        )

        # Prepare the records for Pinecone
        records = []
        for c, e in zip(context, embeddings):
            records.append({
                "id": c['id'],
                "values": e['values'],
                "metadata": {'text': c['text']}
            })

        # Upsert into Pinecone
        index = pc.Index(pinecone_index_name)
        namespace = f"{unique_id}-context"
        index.upsert(namespace=namespace, vectors=records)

        time.sleep(10)

        return jsonify({"message": "Pinecone ingested successfully!", "context": context}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if file is not None:
            # Clean up input and output directories
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
                for output_file in os.listdir(OUTPUT_DIR):
                    os.remove(os.path.join(OUTPUT_DIR, output_file))
        print("Clean up successful!")

# Function to check if a namespace exists in a given index
def namespace_exists(namespace_name):
    # Connect to the specified index
    index = pc.Index(pinecone_index_name)

    # Fetch index description to get metadata (including namespaces)
    index_stats = index.describe_index_stats()

    # Check if the namespace exists in the index metadata
    return namespace_name in index_stats.get("namespaces", {})

# Function to Get OpenAI GPT Response
async def get_gpt_response(value, res=None):
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful sms assistant. Provide clear and concise responses to customer queries. Be professional and conversational. Answer questions based on the context provided."},
                {"role": "user", "content": value if res is None else value + f"Use the following context: {res}"}
            ],
            max_tokens=16384
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        return jsonify({"openai error": "Rate limit exceeded: " + str(e)}), 429
    except OpenAIError as e:
        return jsonify({"openai error": "OpenAI API error: " + str(e)}), 500
    except Exception as e:
        return jsonify({"openai error": str(e)}), 400

@app.route('/message-teli-data', methods=['POST'])
# @require_api_key
async def message_teli_data():
    try:
        # Grab data from the request body
        data = request.get_json()
        if not data:
            return {"error": "Empty or invalid JSON body"}, 400

        unique_id = data.get("unique_id")
        message_history = data.get("message_history")
        stringified = str(message_history)[1:-1]

        last_response = message_history[-1]["message"]

        if not all([unique_id, message_history]):
            return {"error": "Missing required fields"}, 400

        # Check if the namespace exists in the index
        namespace = f"{unique_id}-context"
        if not namespace_exists(namespace):
            return {"error": "Namespace not found in the index"}, 400


        # Convert the last_response into a numerical vector that Pinecone can search with
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[last_response],
            parameters={
                "input_type": "query"
            }
        )

        # # Retrieve the vector embeddings from the index
        index = pc.Index(pinecone_index_name)

        # Search for the most relevant context in the index
        response = index.query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=4,
            include_values=False,
            include_metadata=True
        )

        # Establish score threshold for the highest rated response
        threshold = 0.8
        curr_threshold = response.matches[0].score
        curr_response = response.matches[0].metadata['text']
        if curr_threshold <= threshold:
            gpt_response = await get_gpt_response(stringified)
            return jsonify({"context": gpt_response}), 200

        # Return the most relevant context
        gpt_response = await get_gpt_response(stringified, curr_response)
        return jsonify({"context": gpt_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/delete-namespace', methods=['DELETE'])
# @require_api_key
def delete_namespace():
    try:
        # Grab data from the request body
        data = request.get_json()
        if not data:
            return {"error": "Empty or invalid JSON body"}, 400

        unique_id = data.get("unique_id")

        if not unique_id:
            return {"error": "Missing required fields"}, 400

        # Check if the namespace exists in the index
        namespace = f"{unique_id}-context"
        if not namespace_exists(namespace):
            return {"error": "Namespace not found in the index"}, 400

        # Delete the namespace from the index
        index = pc.Index(pinecone_index_name)
        index.delete(delete_all=True, namespace=namespace)

        return jsonify({"message": "Namespace deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Modal Deployment Functions
@modal_app.function(
        image=image,
        secrets=[Secret.from_name("context-messenger-secrets")]
)
@asgi_app()
def flask_app():
    return app

@modal_app.local_entrypoint()
def serve():
    flask_app.serve()
