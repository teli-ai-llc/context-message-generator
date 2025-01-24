from io import BytesIO
from config import Config
from quart_cors import cors
from functools import wraps
from pydantic import BaseModel
import os, json, time, logging
from quart import Quart, request, jsonify
from openai import RateLimitError, OpenAIError
from redis import Redis, asyncio as aioredis
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalDownloaderConfig,
    LocalConnectionConfig
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

# Connection pooling for Redis
pool = aioredis.from_url(
    os.environ.get("REDIS_URL"),
    max_connections=500,  # Increased max connections for scalability
    socket_timeout=5  # Timeout in seconds
)
redis_client = Redis(connection_pool=pool)


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


async def save_to_redis(file, key, expiration=3600):
    try:
        file_data = await file.read()
        redis_client.set(key, file_data, ex=expiration)  # Store with expiration
    except Exception as e:
        raise RuntimeError(f"Failed to save file to Redis: {str(e)}")


async def get_from_redis(key):
    try:
        file_data = redis_client.get(key)
        if not file_data:
            raise RuntimeError("File not found in Redis or expired.")
        return BytesIO(file_data)  # Return file as a BytesIO stream
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve file from Redis: {str(e)}")


async def delete_from_redis(key):
    try:
        redis_client.delete(key)
    except Exception as e:
        print(f"Failed to delete file from Redis: {str(e)}")


def format_file_for_vectorizing(document_elements, context):
    arr = []
    counter = 1

    # Add extracted data to the array
    if document_elements:
        arr.extend(
            {"id": f"vec{counter + i}", "text": element['text']}
            for i, element in enumerate(document_elements)
        )
        counter += len(document_elements)

    # Add additional context if provided
    if context:
        arr.extend(
            {"id": f"vec{counter + i}", "text": text}
            for i, text in enumerate(context)
        )

    return arr

@quart_app.route('/ingest-teli-data', methods=['POST'])
@require_api_key
async def ingest_teli_data():
    try:
        form_data = await request.form
        file_data = await request.files

        if "unique_id" not in form_data or "file" not in file_data:
            return jsonify({"error": "Missing required fields"}), 400

        # Get the Pinecone client and configuration details
        pc, pinecone_index_name = (
            config_class.pc,
            config_class.PINECONE_INDEX_NAME,
        )

        unique_id = form_data.get("unique_id")
        context_str = form_data.get("context", "")
        file = file_data.get('file')

        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Secure the filename
        filename = file.filename
        redis_key = f"{unique_id}:{filename}"

        # Validate file type
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension != ".pdf":
            return jsonify({"error": "Unsupported file type"}), 400

        # Save file to Redis
        await save_to_redis(file, redis_key)

        # Retrieve file from Redis for processing
        file_stream = await get_from_redis(redis_key)

        # Process the file using Unstructured
        pipeline = Pipeline.from_configs(
            context=ProcessorConfig(),
            downloader_config=LocalDownloaderConfig(),
            source_connection_config=LocalConnectionConfig(),
            partitioner_config=PartitionerConfig(
                partition_by_api=True,
                api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
                partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
                strategy="hi_res",
                additional_partition_args={
                    "split_pdf_page": True,
                    "split_pdf_allow_failed": True,
                    "split_pdf_concurrency_level": 15
                }
            ),
        )

        # Process in-memory file with Unstructured pipeline
        document_elements = pipeline.run(file_stream=file_stream)

        # Parse context string
        context_arr = json.loads(context_str) if context_str else []

        # Format extracted data for vectorizing
        context = format_file_for_vectorizing(document_elements, context_arr)

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
            {
                "id": c['id'],
                "values": e['values'],
                "metadata": {"text": c['text']}
            }
            for c, e in zip(context, embeddings)
        ]

        # Upsert data into Pinecone
        index = pc.Index(pinecone_index_name)
        namespace = f"{unique_id}-context"
        batch = []
        for record in records:
            batch.append(record)
            if len(batch) == batch_size:
                index.upsert(namespace=namespace, vectors=batch)
                batch.clear()

        # Upsert leftover records
        if batch:
            index.upsert(namespace=namespace, vectors=batch)

        # IMPORTANT: Significantly slows down response time, but necessary for the vectorizing process to complete
        time.sleep(10)

        # Cleanup: Delete Redis key
        await delete_from_redis(redis_key)

        return jsonify({"message": "Pinecone ingested successfully!", "context": context}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to check if a namespace exists in a given index
def namespace_exists(namespace_name):
    index = config_class.pc.Index(config_class.PINECONE_INDEX_NAME)
    metadata = index.describe_index_stats().get("namespaces", {})
    return namespace_name in metadata


# Get OpenAI GPT Response
async def get_gpt_response(value, res=None):
    aclient = config_class.aclient

    class Sentiment(BaseModel):
        response: str
        is_conversation_over: str

    try:
        context_message = value if res is None else value + f"Use the following context: {res}"

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful sms assistant. Provide clear and concise responses to customer queries. Be professional and conversational. Answer questions based on the context provided. If the conversation is over, supply a sentiment value of True and False if it's not over"},
                {"role": "user", "content": context_message}
            ],
            response_format=Sentiment,
            max_tokens=16384
        )

        return {**response.choices[0].message.parsed.dict(), **response.usage.dict()}
    except RateLimitError as e:
        return jsonify({"openai error": "Rate limit exceeded: " + str(e)}), 429
    except OpenAIError as e:
        return jsonify({"openai error": "OpenAI API error: " + str(e)}), 500
    except Exception as e:
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
        message_history = str(message_history)

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
        if curr_threshold < threshold:
            gpt_response = await get_gpt_response(message_history)
            if gpt_response.response.is_conversation_over == "True":
                return jsonify({"response": ""}), 200
            return jsonify({"response": gpt_response}), 200

        # Return the most relevant context
        curr_response = response.matches[0].metadata.get('text', '')
        gpt_response = await get_gpt_response(message_history, curr_response)
        if gpt_response.response.is_conversation_over == "True":
            return jsonify({"response": ""}), 200
        return jsonify({"response": gpt_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
