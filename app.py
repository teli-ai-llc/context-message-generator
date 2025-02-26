from config import Config
from quart_cors import cors
from functools import wraps
from pydantic import BaseModel
import os, json, logging, time, dotenv, asyncio
from quart import Quart, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAIError, RateLimitError
from modal import Image, App, Secret, asgi_app, NetworkFileSystem
import sentencepiece as spm
from transformers import AutoTokenizer

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

# Load SentencePiece tokenizer for multilingual-e5-large
sp = spm.SentencePieceProcessor()
TOKENIZER_DIR = "./tokenizer_model"
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "sentencepiece.bpe.model")

# Ensure tokenizer is downloaded
tokenizer = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-large",
    cache_dir=TOKENIZER_DIR,
    use_fast=True  # Ensure fast tokenizer is used
)

if not os.path.exists(TOKENIZER_PATH):
    print(f"Tokenizer model not found at {TOKENIZER_PATH}. Downloading...")
    tokenizer.save_pretrained("./tokenizer_model")
    TOKENIZER_PATH = "./tokenizer_model/sentencepiece.bpe.model"


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
    .pip_install_from_requirements("requirements.txt")
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tokenization Functions
def truncate_text(text, max_tokens=96):
    """Ensure text does not exceed the model's token limit before sending to Pinecone."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    truncated_tokens = tokens[:max_tokens]  # Force truncate to 96 tokens
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    logger.info(f"Truncating: {len(tokens)} tokens -> {len(truncated_tokens)} tokens")
    assert len(truncated_tokens) <= max_tokens, f"Error: Still too long after truncation ({len(truncated_tokens)} tokens)"

    return truncated_text

def chunk_text(text, max_tokens=96):
    """Splits long text into chunks of 96 tokens each."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

async def shorten_text_with_gpt(text):
    """
    Uses GPT to shorten text while maintaining its meaning, ensuring it stays within 96 tokens.
    """
    aclient = Config.aclient

    class ShortenedText(BaseModel):
        shortened: str

    try:
        prompt = (
            "Your task is to shorten the following text while keeping its original meaning. "
            "Ensure the output is 96 tokens or fewer. Do not remove key details. Here is the text:\n\n"
            f"{text}"
        )

        response = await aclient.beta.chat.completions.parse(
            model="gpt-40 mini",
            messages=[{"role": "system", "content": "You are a text summarization expert. Your job is to shorten long texts while maintaining their full meaning. The shortened text should always be 96 tokens or fewer."},
                      {"role": "user", "content": prompt}],
            response_format=ShortenedText,
            max_tokens=96
        )

        shortened_text = response.choices[0].message.parsed.shortened
        token_count = len(tokenizer.encode(shortened_text, add_special_tokens=False))

        if token_count > 96:
            logger.warning(f"GPT Shortened Text is still too long ({token_count} tokens). Further truncation may be needed.")
            shortened_text = truncate_text(shortened_text, max_tokens=96)

        logger.info(f"Shortened Text: {shortened_text} ({token_count} tokens)")
        return shortened_text

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return truncate_text(text, max_tokens=96)  # Fallback to hard truncation
    except Exception as e:
        logger.error(f"Unexpected error in text shortening: {e}")
        return truncate_text(text, max_tokens=96)  # Fallback to hard truncation

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
        filename = file_data['filename'] if file_data else unique_id

        input_endpoint = f"{unique_id}-{filename}"
        input_file_path = os.path.join(INPUT_DIR, input_endpoint)

        if file_data:
            # Save the uploaded file in the input directory
            file_content = file_data['content'].encode("latin1")  # Decode string back to bytes
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

        # Process text before embedding
        text_inputs = []
        for c in context:
            token_count = len(tokenizer.encode(c['text'], add_special_tokens=False))

            if token_count > 96:
                shortened_text = asyncio.run(shorten_text_with_gpt(c['text']))
                text_inputs.extend(chunk_text(shortened_text))
            else:
                text_inputs.append(c['text'])

        for i, text in enumerate(text_inputs):
            length = len(tokenizer.encode(text, add_special_tokens=False))
            logger.info(f"Text {i}: {length} tokens -> {text}")
            assert length <= 96, f"Error: Text chunk {i} is still too long ({length} tokens)"

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

        logger.info("Data ingestion completed successfully!")
        return {"message": "Data ingested successfully!", "context": context}

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
        print(form_data)
        print(file_data)

        if "unique_id" not in form_data or ("file" not in file_data and "context" not in form_data):
            return jsonify({"error": "Missing required fields"}), 400

        unique_id = form_data.get("unique_id")
        context_str = form_data.get("context", "")
        file = file_data.get("file")
        serialized_file = None

        # Serialize file data
        if file:
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
