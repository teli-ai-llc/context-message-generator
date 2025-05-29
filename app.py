from config import Config
from quart_cors import cors
from functools import wraps
from pydantic import BaseModel, Field
import os, json, logging, time, dotenv, asyncio
from quart import Quart, request, jsonify
from aiohttp import ClientSession
from werkzeug.utils import secure_filename
from openai import OpenAIError, RateLimitError
from modal import Image, App, Secret, asgi_app, NetworkFileSystem
import sentencepiece as spm
from transformers import AutoTokenizer

# For testing purposes
from loan_data import loan_data  # Assuming loan_data is a dictionary with the required structure
from lead_data import lead_data  # Assuming lead_data is a dictionary with the required structure

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
        batch_size = 96

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

        if "unique_id" not in form_data or "file" not in file_data:
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

class SchemaDiff(BaseModel):
    updated_schema: dict = Field(..., description="The updated version of the input schema")
    changes: dict = Field(..., description="The differences found from comparing user input")

    model_config = {
        "json_schema_extra": {
            "required": ["updated_schema", "changes"]
        }
    }

class Sentiment(BaseModel):
    response: str
    conversation_status: str

async def get_lead_info(lead_id, loan_id):
    """
    Fetches lead and loan information from the Lodasoft API.
    Returns a tuple of (lead_data, loan_application_model).
    """

    async with ClientSession() as session:
        async with session.get(f"https://publicapi.lodasoft.com/api/leads/{lead_id}") as response:
            response.raise_for_status()
            lead_data = await response.json()
            logger.info(f"Fetched lead data for lead_id: {lead_id}")

        async with session.get(f"https://publicapi.lodasoft.com/api/Loan/{loan_id}/get-application-model") as response:
            response.raise_for_status()
            loan_application_model = await response.json()
            logger.info(f"Fetched loan application model for loan_id: {loan_id}")

    return lead_data, loan_application_model

async def gpt_schema_update(aclient, schema_type: str, original: dict, message_history: str) -> tuple[dict, dict]:
    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a data assistant. The following is the original {schema_type} schema. "
                "Compare it to the information the user provides in the conversation and return:\n"
                '{ "updated_schema": {...}, "changes": { "field": { "old": ..., "new": ... } } }'
                "\nRespond ONLY with a JSON object. Do not include ```json or any explanation."
            )
        },
        {
            "role": "user",
            "content": json.dumps({
                "original_schema": original,
                "conversation": message_history
            })
        }
    ]

    response = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        max_tokens=16384
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown code fences like ```json ... ```
    if raw.startswith("```json"):
        raw = raw.removeprefix("```json").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
        print(parsed)
        return parsed.get("updated_schema", {}), parsed.get("changes", {})
    except json.JSONDecodeError:
        logger.error("GPT response was not valid JSON:\n" + raw)
        return {}, {}

async def gpt_response(message_history, retrieved_contexts=None, goal=None, lead_id=None, loan_id=None):
    aclient = Config.aclient

    # lead_data, loan_application_model = await get_lead_info(lead_id, loan_id) if lead_id and loan_id else (None, None)

    try:

        updated_lead_data, lead_changes = await gpt_schema_update(aclient, "lead", lead_data or {}, message_history)
        updated_loan_data, loan_changes = await gpt_schema_update(aclient, "loan", loan_data or {}, message_history)
        # updated_loan_data, loan_changes = await gpt_schema_update(aclient, "loan", loan_application_model or {}, message_history)

        change_log = ""
        for field, diff in {**lead_changes, **loan_changes}.items():
            change_log += f"- `{field}` changed from `{diff['old']}` to `{diff['new']}`\n"

        # Use the top 5 retrieved contexts (already ranked)
        if retrieved_contexts and isinstance(retrieved_contexts, list):
            top_contexts = [ctx["text"] for ctx in retrieved_contexts[:5]]
            ranked_context_str = "\n\n".join(top_contexts) if top_contexts else None
        else:
            ranked_context_str = None

        # Construct final context message
        context_message = message_history
        if ranked_context_str:
            context_message += (
                f"\n\nNote: The following relevant information has been retrieved. "
                "Use this as a reference before determining if further clarification is needed.\n"
                f"{ranked_context_str}"
            )
        else:
            context_message += (
                "\n\nNote: No relevant information was found in prior records. "
                "If the question is unrelated to the topic, politely guide the conversation back on track."
            )

        if change_log:
            context_message += f"\n\nThe following updates were inferred from the conversation:\n{change_log}"

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide clear, professional, and helpful responses in a conversational tone. "
                        "Ensure accuracy while keeping interactions natural and engaging.\n\n"

                        f"The goal of this conversation is: {goal}\n\n"

                        "**Guidelines for Handling Conversations:**\n"
                        "- **conversation_over** → Use this only if the user clearly states they have no further questions.\n"
                        "- **human_intervention** → Escalate only if the user asks about scheduling, availability, or if no clear answer is found in the provided context.\n"
                        "- **continue_conversation** → If the topic allows for further discussion, offer additional insights or ask if the user would like more details.\n"
                        "- **out_of_scope** → If the user's question is unrelated, acknowledge it politely and redirect the conversation back to relevant topics.\n\n"

                        "**Handling Out-of-Scope Questions:**\n"
                        "If a user asks something unrelated, respond in a way that maintains a natural flow:\n"
                        "👤 User: 'What's the best Italian restaurant nearby?'\n"
                        "💬 Response: 'That sounds like a great topic! While I don't have restaurant recommendations, I'd be happy to assist with [specific topic]. Let me know how I can help! 😊'\n\n"

                        "If the user continues with off-topic questions, acknowledge their curiosity but steer the conversation back in a professional and engaging manner."
                    )
                },
                {"role": "user", "content": context_message}
            ],
            response_format=Sentiment,
            max_tokens=16384
        )

        parsed_sentiment = response.choices[0].message.parsed
        token_usage = response.usage.to_dict()

        # If GPT determines the question is off-topic, classify as 'out_of_scope'
        if "I'm not sure" in parsed_sentiment.response or "I can't help with that" in parsed_sentiment.response:
            parsed_sentiment.conversation_status = "out_of_scope"

        return {
            "response": parsed_sentiment.response,
            "conversation_status": parsed_sentiment.conversation_status,  # Tracks 'conversation_over', 'human_intervention', 'continue_conversation', 'out_of_scope'
            "updated_lead_data": updated_lead_data,
            "updated_loan_application_model": updated_loan_data,
            "changes": {
                "lead_changes": lead_changes,
                "loan_changes": loan_changes
            },
            **token_usage
        }

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
        # loan_id = data.get("loan_id", None)
        # lead_id = data.get("lead_id", None)
        goal = data.get("goal", None)

        if not all([unique_id, message_history]):
            return jsonify({"error": "Missing required fields"}), 400

        last_message = message_history[-1]["message"]
        stringified_message_history = str(message_history)

        # Check if the namespace exists in the index
        namespace = f"{unique_id}-context"
        if not namespace_exists(namespace):
            logger.info(f"Namespace {namespace} does not exist")
            return jsonify({"error": "Namespace not found in the index"}), 400

        # Convert the last_response into a numerical vector for Pinecone search
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[last_message],
            parameters={"input_type": "query"}
        )

        # Retrieve relevant context from Pinecone
        index = pc.Index(pinecone_index_name)
        response = index.query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=5,  # Fetch more results to rank them
            include_values=False,
            include_metadata=True
        )

        # Rank the retrieved contexts by their score (highest first)
        if response.matches:
            ranked_contexts = sorted(
                response.matches, key=lambda x: x.score, reverse=True
            )
            top_contexts = [{"text": match.metadata.get("text", ""), "score": match.score} for match in ranked_contexts]
        else:
            top_contexts = []

        # Establish score threshold to determine if context is useful
        threshold = 0.8
        if not top_contexts or top_contexts[0]["score"] < threshold:
            logger.info("No highly relevant context found. Using GPT alone.")
            gpt_response_data = await gpt_response(stringified_message_history, goal=goal)
        else:
            logger.info("Using ranked context for GPT response.")
            gpt_response_data = await gpt_response(stringified_message_history, top_contexts, goal=goal)

        # Handle response based on conversation status
        conversation_status = gpt_response_data["conversation_status"]
        # response_text = gpt_response_data["response"]
        response_text = gpt_response_data

        if conversation_status == 'human_intervention':
            logger.info("Human intervention required")
            return jsonify({"response": "Human intervention required"}), 200
        elif conversation_status == 'conversation_over':
            logger.info("Conversation complete")
            return jsonify({"response": "Conversation complete"}), 200
        elif conversation_status == 'continue_conversation' or conversation_status == 'out_of_scope':
            logger.info("Continue conversation")
            return jsonify({"response": response_text}), 200

    except Exception as e:
        logger.error(f"Error generating response: {e}")
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
