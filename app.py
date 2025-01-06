import os, json
from config import Config
from quart import Quart, request, jsonify
from quart_cors import cors
from werkzeug.utils import secure_filename
import time
from functools import wraps
from modal import Image, App, Secret, asgi_app
from openai import RateLimitError, OpenAIError

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

quart_app = Quart(__name__)
quart_app = cors(
    quart_app,
    allow_origin="*",
    allow_headers="*",
    allow_methods=["POST", "DELETE"]
)

# Create a Modal App and Image with the required dependencies
modal_app = App("context-message-generator")

image = (
    Image.debian_slim()
    .apt_install("curl", "git", "protobuf-compiler")  # Install dependencies
    .run_commands(
        [
            "curl -OL https://go.dev/dl/go1.20.8.linux-amd64.tar.gz",
            "tar -C /usr/local -xzf go1.20.8.linux-amd64.tar.gz",
            "rm go1.20.8.linux-amd64.tar.gz",
            "ln -s /usr/local/go/bin/go /usr/bin/go",
            "ln -s /usr/local/go/bin/gofmt /usr/bin/gofmt",
            # Install protoc-gen-openapiv2
            "go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2@v2.14.0",
            "cp $(go env GOPATH)/bin/protoc-gen-openapiv2 /usr/local/bin",
            # Verify installation
            "protoc-gen-openapiv2 --version || echo 'Installation failed'",
        ]
    )
    .pip_install_from_requirements("requirements.txt")  # Install Python dependencies
)

# Initialize the Config class
Config.initialize()

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per chunk

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

    OUTPUT_DIR = Config.OUTPUT_DIR

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

@quart_app.route('/ingest-teli-data', methods=['POST'])
# @require_api_key
async def ingest_teli_data():
    try:
        form_data = await request.form
        file_data = await request.files

        if "unique_id" not in form_data and ("file" not in file_data or "context" not in file_data):
            return jsonify({"error": "Missing required fields"}), 400

        # Get the Pinecone client
        pc = Config.pc
        INPUT_DIR = Config.INPUT_DIR
        OUTPUT_DIR = Config.OUTPUT_DIR
        pinecone_index_name = Config.PINECONE_INDEX_NAME
        unstructured_api_key = Config.UNSTRUCTURED_API_KEY
        unstructured_api_url = Config.UNSTRUCTURED_API_URL

        unique_id = form_data.get("unique_id")
        context_str = form_data.get("context")
        file = file_data.get('file')

        # Parse context string
        context_arr = json.loads(context_str) if context_str else None

        filename = None
        file_extension = None
        input_endpoint = None
        input_file_path = None

        if file is not None:
            # Secure the filename
            filename = secure_filename(file.filename)
            file_extension = os.path.splitext(filename)[1].lower()
            input_endpoint = f"{unique_id}-{filename}"

            if file_extension not in [".pdf"]:
                return jsonify({"error": "Unsupported file type"}), 400

            # Save the uploaded file in the input directory in chunks
            input_file_path = os.path.join(INPUT_DIR, input_endpoint)
            with open(input_file_path, "wb") as f:
                while True:
                    chunk = file.stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)

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
            print("Pipeline ran successfully!")

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

        # IMPORTANT: Significantly slows down response time, but necessary for the data to be available for querying
        time.sleep(10)

        if file:
            # Clean up input and output directories
            if input_file_path and os.path.exists(input_file_path):
                os.remove(input_file_path)
                for output_file in os.listdir(OUTPUT_DIR):
                    output_file_path = os.path.join(OUTPUT_DIR, output_file)
                    if os.path.exists(output_file_path):
                        os.remove(output_file_path)
                print("Clean up successful!")

        return jsonify({"message": "Pinecone ingested successfully!", "context": context}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to check if a namespace exists in a given index
def namespace_exists(namespace_name):
    # Connect to the specified index
    pc = Config.pc
    index = pc.Index(Config.PINECONE_INDEX_NAME)

    # Fetch index description to get metadata (including namespaces)
    index_stats = index.describe_index_stats()

    # Check if the namespace exists in the index metadata
    return namespace_name in index_stats.get("namespaces", {})

# Function to Get OpenAI GPT Response
async def get_gpt_response(value, res=None):
    try:
        aclient = Config.aclient

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

@quart_app.route('/message-teli-data', methods=['POST'])
# @require_api_key
async def message_teli_data():
    try:
        # Grab data from the request body
        data = await request.json

        # Get the Pinecone client
        pc = Config.pc
        pinecone_index_name = Config.PINECONE_INDEX_NAME

        unique_id = data.get("unique_id")
        message_history = data.get("message_history")
        stringified = str(message_history)

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

@quart_app.route('/delete-namespace', methods=['DELETE'])
# @require_api_key
def delete_namespace():
    try:
        # Get the Pinecone client
        pc = Config.pc

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
        index = pc.Index(Config.get('PINECONE_INDEX_NAME'))
        index.delete(delete_all=True, namespace=namespace)

        return jsonify({"message": "Namespace deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# For deployment with Modal
@modal_app.function(
    image=image,
    secrets=[Secret.from_name("context-messenger-secrets")]
)
@asgi_app()
def quart_asgi_app():
    return quart_app

# Local entrypoint for running the app
@modal_app.local_entrypoint()
def serve():
    quart_app.run()
