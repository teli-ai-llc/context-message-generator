from io import BytesIO
from config import Config
from quart_cors import cors
from functools import wraps
import os, logging, asyncio
from pydantic import BaseModel
from quart import Quart, request, jsonify
from openai import RateLimitError, OpenAIError
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.embedder import EmbedderConfig
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured_ingest.connector.pinecone import PineconeDestination, PineconeConfig

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Semaphore to limit concurrency
semaphore = asyncio.Semaphore(500)

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


async def process_with_limit(func, *args, **kwargs):
    async with semaphore:
        return await func(*args, **kwargs)


@quart_app.route('/ingest-teli-data', methods=['POST'])
@require_api_key
async def ingest_teli_data():
    try:
        form_data = await request.form
        file_data = await request.files

        if "unique_id" not in form_data or "file" not in file_data:
            return jsonify({"error": "Missing required fields"}), 400

        unique_id = form_data.get("unique_id")
        file = file_data.get('file')

        if not file:
            return jsonify({"error": "No file provided"}), 400

        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension != ".pdf":
            return jsonify({"error": "Unsupported file type"}), 400

        file_content = await file.read()

        # Process the file with concurrency control
        await process_with_limit(process_pipeline, unique_id, file_content)

        return jsonify({"message": "File processed successfully."}), 200

    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        return jsonify({"error": str(e)}), 500

async def process_pipeline(unique_id, file_content):
    # Configure the pipeline
    pipeline = Pipeline.from_configs(
        context=ProcessorConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
            strategy="hi_res",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
            },
        ),
        embedder_config=EmbedderConfig(
            embed_by_api=True,
            api_key=os.environ.get("EMBEDDING_API_KEY"),
            embed_endpoint=os.environ.get("EMBEDDING_API_URL"),
            model_name="text-embedding-ada-002",
        ),
        chunker_config=ChunkerConfig(
            chunking_strategy="by_title",
            max_characters=1000,
            overlap=50,
        ),
    )

    # Configure Pinecone destination
    pinecone_config = PineconeConfig(
        api_key=os.environ.get("PINECONE_API_KEY"),
        index_name=os.environ.get("PINECONE_INDEX_NAME"),
        namespace=f"{unique_id}-namespace"
    )
    destination = PineconeDestination(config=pinecone_config)

    # Process the file
    pipeline.run(source=BytesIO(file_content), destination=destination)


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
