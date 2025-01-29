from config import Config
from quart_cors import cors
from functools import wraps
from pydantic import BaseModel
import os, json, logging, time, dotenv
from quart import Quart, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAIError, RateLimitError
from modal import Image, App, Secret, asgi_app, NetworkFileSystem

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

# Load the configuration and initialize env variables
quart_app.config["APP_CONFIG"] = Config()
config_class = quart_app.config["APP_CONFIG"]
config_class.initialize()

# Create a Modal App and Network File System
modal_app = App("context-message-generator")
network_file_system = NetworkFileSystem.from_name("context-message-generator-nfs", create_if_missing=True)

image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")  # Install Python dependencies
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def format_file_for_vectorizing(file, context, endpoint):
    arr = []
    OUTPUT_DIR = config_class.OUTPUT_DIR

    counter = 1

    if file:
        with open(os.path.join(OUTPUT_DIR, f"{endpoint}.json"), "r") as f:
            content = json.load(f)
            arr.extend(
                {"id": f"vec{counter + i}", "text": obj["text"]}
                for i, obj in enumerate(content)
            )
        counter += len(content)

    if context:
        arr.extend(
            {"id": f"vec{counter + i}", "text": text}
            for i, text in enumerate(context)
        )

    logger.info(f"Formatted context for vectorizing")
    return arr

@modal_app.function(
    network_file_systems={"/uploads": network_file_system},
    image=image,
    secrets=[Secret.from_name("context-messenger-secrets")]
)
def vectorize(unique_id, context_str, file_data):
    try:
        # Get the Pinecone client and configuration details
        pc, INPUT_DIR, OUTPUT_DIR, pinecone_index_name, unstructured_api_key, unstructured_api_url = (
            config_class.pc,
            config_class.INPUT_DIR,
            config_class.OUTPUT_DIR,
            config_class.PINECONE_INDEX_NAME,
            config_class.UNSTRUCTURED_API_KEY,
            config_class.UNSTRUCTURED_API_URL,
        )

        context_arr = json.loads(context_str) if context_str else []
        filename = file_data['filename']
        file_content = file_data['content'].encode("latin1")  # Decode string back to bytes

        input_endpoint = f"{unique_id}-{filename}"
        input_file_path = os.path.join(INPUT_DIR, input_endpoint)

        if file_data:
            # Save the uploaded file in the input directory
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension != ".pdf":
                return jsonify({"error": "Unsupported file type"}), 400

            # Save the uploaded file in the input directory in chunks
            input_endpoint = f"{unique_id}-{filename}"
            input_file_path = os.path.join(INPUT_DIR, input_endpoint)

            with open(input_file_path, "wb") as f:
                f.write(file_content)

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
            logger.info("Pipeline ran successfully!")

        # Read processed output files
        context = format_file_for_vectorizing(file_data, context_arr, input_endpoint)

        # Handle embedding batch sizes to avoid memory issues and input size limits
        batch_size = 100
        text_inputs = [c['text'] for c in context]
        embeddings = []
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i:i + batch_size]
            batch_embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch,
                parameters={
                    "input_type": "passage",
                    "truncate": "END",
                },
            )
            embeddings.extend(batch_embeddings)

        # Prepare the records
        records = [
            {"id": c['id'], "values": e['values'], "metadata": {"text": c['text']}}
            for c, e in zip(context, embeddings)
        ]

        # Upsert records in batches to avoid memory issues and input size limits
        index = pc.Index(pinecone_index_name)
        namespace = f"{unique_id}-context"
        for i in range(0, len(records), batch_size):
            index.upsert(namespace=namespace, vectors=records[i : i + batch_size])

        # Wait for the index to update
        while True:
            stats = index.describe_index_stats()
            current_vector_count = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
            if current_vector_count >= len(records):
                break
            time.sleep(1)  # Wait and re-check periodically

        if file_data:
            try:
                os.remove(input_file_path)
                logger.info(f"Input file {input_file_path} deleted successfully.")

                for output_file in os.listdir(OUTPUT_DIR):
                    os.remove(os.path.join(OUTPUT_DIR, output_file))
                    logger.info("Output directory cleaned successfully.")
            except Exception as e:
                logger.error(f"Error deleting input file: {e}")
                return {"error": "Error deleting input file"}

        logger.info("Pinecone ingestion completed successfully!")
        return {"message": "Pinecone ingested successfully!", "context": context}

    except Exception as e:
        logger.info(f"Error ingesting data: {e}")
        return {"error": str(e)}

@quart_app.route('/ingest-teli-data', methods=['POST'])
@require_api_key
async def ingest_teli_data():
    try:
        # Extract form data and file
        form_data = await request.form
        file_data = await request.files

        if "unique_id" not in form_data or "file" not in file_data:
            return jsonify({"error": "Missing required fields"}), 400

        unique_id = form_data.get("unique_id")
        context_str = form_data.get("context", "")
        file = file_data.get("file")

        # Serialize file data
        file_content = file.read()  # Read the file as bytes
        serialized_file = {
            "filename": secure_filename(file.filename),
            "content": file_content.decode("latin1"),  # Encode bytes to string for transmission
        }

        # Call the Modal function
        result = vectorize.remote(unique_id, context_str, serialized_file)

        if "error" in result:
            logger.info(f"Error in Quart endpoint: {result['error']}")
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in Quart endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Function to check if a namespace exists in a given index
def namespace_exists(namespace_name):
    index = config_class.pc.Index(config_class.PINECONE_INDEX_NAME)
    metadata = index.describe_index_stats().get("namespaces", {})
    return namespace_name in metadata

async def get_gpt_response(message_history, res=None):
    aclient = Config.aclient

    class Sentiment(BaseModel):
        response: str
        is_conversation_over: bool

    try:
        context_message = message_history if res is None else message_history + f"Use the following context: {res}"

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful SMS assistant. Your job is to provide clear and concise responses to customer queries in a professional and conversational tone using the provided context\n\n"
                                "In addition to responding to the user, determine whether the conversation has reached a conclusion or requires human intervention.\n\n"
                                "**Guidelines for `is_conversation_over`:**\n"
                                    "- **Set to `True`** if:\n"
                                    "  - The user's query has been fully addressed, and no further questions are expected.\n"
                                    "  - The user requests human support or expresses frustration.\n"
                                    "  - The issue is too complex for automation.\n\n"
                                    "- **Set to `False`** if:\n"
                                    "  - The user is likely to continue the conversation (e.g., asking for more details or clarification).\n"
                                    "  - There is an open-ended discussion that requires further engagement.\n\n"
                                    "When in doubt, assume that the user may have further questions and set `is_conversation_over: False`.\n\n"
                                    "Return your response using the following structured format:\n"
                                    "`Sentiment(response='<Your response to the user>', is_conversation_over=<True or False>)`"
                },
                {"role": "user", "content": context_message}
            ],
            response_format=Sentiment,
            max_tokens=16384
        )

        # Access the parsed Sentiment object directly
        parsed_sentiment = response.choices[0].message.parsed
        token_usage = response.usage.dict()

        # Create a dictionary with the response and token usage
        res_dict = {
            "response": parsed_sentiment.response,
            "is_conversation_over": parsed_sentiment.is_conversation_over,
            **token_usage
        }

        logger.info("Response Generated Successfully!")

        return res_dict
    except RateLimitError as e:
        logger.info(f"Rate limit exceeded: {e}")
        return jsonify({"openai error": "Rate limit exceeded: " + str(e)}), 429
    except OpenAIError as e:
        logger.info(f"OpenAI API error: {e}")
        return jsonify({"openai error": "OpenAI API error: " + str(e)}), 500
    except Exception as e:
        logger.info(f"Error generating response: {e}")
        return jsonify({"openai error": str(e)}), 400


@quart_app.route('/message-teli-data', methods=['POST'])
@require_api_key
async def message_teli_data():
    try:
        # Grab data from the request body
        data = await request.json

        # Get the Pinecone client
        pc, pinecone_index_name = config_class.pc, config_class.PINECONE_INDEX_NAME

        unique_id = data.get("unique_id")
        message_history = data.get("message_history")
        stringified_message_history = str(message_history)

        last_message = message_history[-1]["message"]

        if not all([unique_id, message_history]):
            return {"error": "Missing required fields"}, 400

        # Check if the namespace exists in the index
        namespace = f"{unique_id}-context"
        if not namespace_exists(namespace):
            logger.info(f"Namespace {namespace} does not exist")
            return {"error": "Namespace not found in the index"}, 400


        # Convert the last_response into a numerical vector that Pinecone can search with
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[last_message],
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
        if curr_threshold < threshold:
            gpt_response = await get_gpt_response(stringified_message_history)
            if gpt_response["is_conversation_over"] == True:
                logger.info(f"Conversation completed")
                return jsonify({"response": "Conversation completed"}), 200
            return jsonify({"response": gpt_response["response"]}), 200

        # Return the most relevant context
        curr_response = response.matches[0].metadata.get('text', '')
        print(stringified_message_history)
        gpt_response = await get_gpt_response(stringified_message_history, curr_response)
        if gpt_response["is_conversation_over"] == True:
            logger.info(f"Conversation completed")
            return jsonify({"response": "Conversation completed"}), 200
        return jsonify({"response": gpt_response["response"]}), 200

    except Exception as e:
        logger.info(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 400

@quart_app.route('/delete-namespace/<unique_id>', methods=['DELETE'])
@require_api_key
async def delete_namespace(unique_id):
    try:
        # Get the Pinecone client
        pc, pinecone_index_name = config_class.pc, config_class.PINECONE_INDEX_NAME

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
        logger.info(f"Error deleting namespace: {e}")
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
