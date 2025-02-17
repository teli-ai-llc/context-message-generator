from config import Config
from quart_cors import cors
from functools import wraps
from pydantic import BaseModel
import os, json, logging, time, aiofiles, asyncio, dotenv
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
async def vectorize(unique_id, context_str, file_data):
    try:
        pc, INPUT_DIR, OUTPUT_DIR, pinecone_index_name, unstructured_api_key, unstructured_api_url = (
            config_class.pc,
            config_class.INPUT_DIR,
            config_class.OUTPUT_DIR,
            config_class.PINECONE_INDEX_NAME,
            config_class.UNSTRUCTURED_API_KEY,
            config_class.UNSTRUCTURED_API_URL,
        )

        # Ensure `context_str` is a list
        try:
            context_arr = json.loads(context_str)
            if not isinstance(context_arr, list):
                raise ValueError("context_str must be a list")
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in context_str"}

        filename = file_data["filename"]
        file_content = file_data["content"].encode("latin1")

        input_endpoint = f"{unique_id}-{filename}"
        input_file_path = os.path.join(INPUT_DIR, input_endpoint)

        if file_data:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension != ".pdf":
                return {"error": "Unsupported file type"}

            async with aiofiles.open(input_file_path, "wb") as f:
                await f.write(file_content)

            await asyncio.to_thread(Pipeline.from_configs(
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
            ).run)
            logger.info("Pipeline ran successfully!")

        context = format_file_for_vectorizing(file_data, context_arr, input_endpoint)

        # Parallel embedding processing
        batch_size = 100
        text_inputs = [c['text'] for c in context]

        async def embed_batch(batch):
            return pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"}
            )

        embedding_tasks = [embed_batch(text_inputs[i:i+batch_size]) for i in range(0, len(text_inputs), batch_size)]
        embeddings = await asyncio.gather(*embedding_tasks)

        # Flatten embeddings
        embeddings = [e for batch in embeddings for e in batch]

        #  Parallel Upsert to Pinecone
        index = pc.Index(pinecone_index_name)
        namespace = f"{unique_id}-context"

        async def upsert_batch(records):
            if records:
                index.upsert(namespace=namespace, vectors=records)

        upsert_tasks = []
        for i, e in enumerate(embeddings):
            if i < len(context) and isinstance(context[i], dict):  # Ensure safe indexing
                upsert_tasks.append(upsert_batch([{
                    "id": context[i]["id"],
                    "values": e["values"],
                    "metadata": {"text": context[i]["text"]}
                }]))

        await asyncio.gather(*upsert_tasks)

        # Asynchronous file cleanup
        async def cleanup_files():
            try:
                await asyncio.to_thread(os.remove, input_file_path)
                for output_file in os.listdir(OUTPUT_DIR):
                    await asyncio.to_thread(os.remove, os.path.join(OUTPUT_DIR, output_file))
                logger.info("Files cleaned successfully.")
            except Exception as e:
                logger.error(f"Error deleting files: {e}")

        await cleanup_files()

        logger.info("Pinecone ingestion completed successfully!")
        return {"message": "Pinecone ingested successfully!", "context": context}

    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
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
        result = vectorize.spawn(unique_id, context_str, serialized_file).get()

        if "error" in result:
            logger.error(f"Error in Modal function: {result['error']}")
            return jsonify(result), 500

        return jsonify({"message": "Vectorization complete!", "task_id": unique_id}), 202

    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
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
        conversation_over_or_human_intervention: str  # New field for unified status

    try:
        context_message = message_history if res is None else message_history + f"\n\nUse the following context: {res}"

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful SMS assistant. Your job is to provide clear and concise responses to customer queries in a professional and conversational tone using the provided context.\n\n"
                                "In addition to responding to the user, determine whether the conversation has reached a conclusion or requires human intervention.\n\n"
                                "**Guidelines for `conversation_over_or_human_intervention`:**\n"
                                    "- **Set to `'conversation_over'`** if:\n"
                                    "  - The user's query has been fully addressed, and no further questions are expected **AND** they indicate closure (e.g., 'Thanks, that's all!').\n"
                                    "  - The conversation naturally concludes without requiring further discussion.\n\n"
                                    "- **Set to `'human_intervention'`** if:\n"
                                    "  - The user requests to speak with someone (e.g., 'Can I talk to a person?', 'Can I get a call?').\n"
                                    "  - The issue is too complex for automation.\n"
                                    "  - The user expresses frustration or dissatisfaction.\n\n"
                                    "- **Set to `'continue_conversation'`** if:\n"
                                    "  - The user is likely to ask follow-ups (e.g., 'Are ocean levels rising?' → They may want details on causes, effects, or solutions).\n"
                                    "  - The response can reasonably prompt further discussion.\n\n"
                                    "**Response Format:**\n"
                                    "{ \"response\": \"<Your response to the user>\", \"conversation_over_or_human_intervention\": \"<conversation_over | human_intervention | continue_conversation>\" }"
                },
                {"role": "user", "content": context_message}
            ],
            response_format=Sentiment,
            max_tokens=1024  # Optimized to prevent token overflow
        )

        # Extract response text
        parsed_sentiment = response.choices[0].message.parsed
        token_usage = response.usage.to_dict()

        # Create a dictionary with the response and token usage
        res_dict = {
            "response": parsed_sentiment.response,
            "conversation_over_or_human_intervention": parsed_sentiment.conversation_over_or_human_intervention,
            **token_usage
        }

        logger.info("Response Generated Successfully!")

        return res_dict
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e}")
        return {"error": "Rate limit exceeded", "message": str(e)}, 429
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return {"error": "OpenAI API error", "message": str(e)}, 500
    except json.JSONDecodeError:
        logger.error("Error decoding OpenAI response as JSON")
        return {"error": "Invalid response format from OpenAI"}, 500
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"error": "Unexpected error", "message": str(e)}, 400


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
            if gpt_response["conversation_over_or_human_intervention"] == 'human_intervention':
                logger.info(f"Human intervention required")
                return jsonify({"response": "Human intervention required"}), 200
            elif gpt_response["conversation_over_or_human_intervention"] == 'conversation_over':
                logger.info(f"Conversation complete")
                return jsonify({"response": "Conversation complete"}), 200
            elif gpt_response["conversation_over_or_human_intervention"] == 'continue_conversation':
                logger.info(f"Continue conversation")
                return jsonify({"response": gpt_response["response"]}), 200

        # Return the most relevant context
        curr_response = response.matches[0].metadata.get('text', '')
        gpt_response = await get_gpt_response(stringified_message_history, curr_response)
        if gpt_response["conversation_over_or_human_intervention"] == 'human_intervention':
            logger.info(f"Human intervention required")
            return jsonify({"response": "Human intervention required"}), 200
        elif gpt_response["conversation_over_or_human_intervention"] == 'conversation_over':
            logger.info(f"Conversation complete")
            return jsonify({"response": "Conversation complete"}), 200
        elif gpt_response["conversation_over_or_human_intervention"] == 'continue_conversation':
            logger.info(f"Continue conversation")
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
