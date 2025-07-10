from quart_cors import cors
from functools import wraps
from decimal import Decimal
from pydantic import BaseModel, Field
import os, json, logging, time, dotenv, asyncio
from quart import Quart, request, jsonify
from aiohttp import ClientSession
from openai import OpenAIError, RateLimitError, AsyncOpenAI
from modal import Image, App, Secret, asgi_app

from repository.context import MessageContext

quart_app = Quart(__name__)
quart_app = cors(
    quart_app,
    allow_origin="*",
    allow_headers="*",
    allow_methods=["POST", "DELETE"]
)


# Create a Modal App and Network File System
modal_app = App("lodasoft-app")
aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

context_store = MessageContext()

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

@quart_app.route("/upload_context", methods=["POST"])
@require_api_key
async def upload_context():
    try:
        data = await request.json
        id = data.get("id")
        context = data.get("context")
        schema_context = data.get("schema_context", [])

        # Validation for the required fields
        if not id or not context or not isinstance(context, list):
            return jsonify({"error": "Missing required fields: id and context are required."}), 400

        if schema_context and not isinstance(schema_context, list):
            return jsonify({"error": "Invalid schema_context format. It must be a list of schemas."}), 400

        # Save context and schema_context to the database
        await context_store.update_message_context(id, context, schema_context)

        logging.info(f"Context uploaded successfully for id {id}.")
        return jsonify({"message": "Context uploaded successfully.", "id": id, "context": context, "schema_context": schema_context}), 200

    except Exception as e:
        logging.error(f"Error uploading context: {e}")
        return jsonify({"error": f"Error uploading context: {str(e)}"}), 500


class SchemaDiff(BaseModel):
    updated_schema: dict = Field(..., description="The updated version of the input schema")
    changes: dict = Field(..., description="The differences found from comparing user input")

    model_config = {
        "json_schema_extra": {
            "required": ["changes"]
        }
    }

class Sentiment(BaseModel):
    response: str
    conversation_status: str

@quart_app.route("/get_context/<id>", methods=["GET"])
@require_api_key
async def get_context(id):
    try:
        if not id:
            return jsonify({"error": "Missing required field: id"}), 400

        # Retrieve context from DynamoDB
        context = context_store.get(id)

        if not context:
            return jsonify({"error": "No context found for the given id"}), 404

        logging.info(f"Context retrieved successfully for id {id}.")
        return jsonify({"id": id, "context": context}), 200

    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        return jsonify({"error": f"Error retrieving context: {str(e)}"}), 500

@quart_app.route("/delete_context/<id>", methods=["DELETE"])
@require_api_key
async def delete_context(id):
    try:
        if not id:
            return jsonify({"error": "Missing required field: id"}), 400

        # Delete context from DynamoDB
        context_store.delete(id)

        logging.info(f"Context deleted successfully for id {id}.")
        return jsonify({"message": "Context deleted successfully.", "id": id}), 200

    except Exception as e:
        logging.error(f"Error deleting context: {e}")
        return jsonify({"error": f"Error deleting context: {str(e)}"}), 500

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float for JSON serialization
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

async def gpt_schema_update(schema_name, original_schema, user_message, message_history):
    try:
        # Create the prompt for GPT to extract schema changes, now including message history
        context_message = "".join([f"{msg['role']}: {msg['message']}\n" for msg in message_history])

        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a data assistant. The following is the original {schema_name} schema. "
                    "The user has provided a message that may contain updates to some fields in the schema.\n\n"
                    "Your task is to extract ONLY the fields clearly mentioned in the message and return them as a nested object.\n"
                    "- If a field is mentioned, include it in the result with its new value.\n"
                    "- If a field is not mentioned, do not include it.\n"
                    "- If the field is nested (inside another object), return it as a nested object.\n\n"
                    "Return ONLY the following JSON structure:\n"
                    '{ \"extracted_fields\": { \"field_name\": value, ... } }\n\n'
                    "Do not include the entire updated schema. Only include the fields mentioned in the user message.\n"
                    "Respond ONLY with a JSON object."
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "original_schema": original_schema,
                    "user_message": user_message,
                    "context_message": context_message
                }, default=decimal_default)  # Ensure Decimal is serialized correctly
            }
        ]

        # Send the request to GPT-4 to extract changes
        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=16384
        )

        # Clean and parse the response from GPT
        raw = response.choices[0].message.content.strip()

        # Check and remove any unwanted markdown code fences
        if raw.startswith("```json"):
            raw = raw.removeprefix("```json").strip()
        if raw.endswith("```"):
            raw = raw.removesuffix("```").strip()

        # Debug: Log the raw response for further analysis
        logger.debug(f"Raw GPT response for {schema_name}: {raw}")

        try:
            # Attempt to parse the cleaned-up response as JSON
            parsed = json.loads(raw)
            logger.debug(f"Parsed GPT response for {schema_name}: {parsed}")

            # Ensure the changes are properly extracted and returned
            extracted_fields = parsed.get("extracted_fields", {})

            # If no fields were extracted, return an empty response for this schema
            if not extracted_fields:
                logger.info(f"No fields extracted for schema {schema_name}.")
                return {}, response.usage.to_dict()

            # Validate fields against the schema to ensure only existing fields are included
            valid_fields = {}
            for field, value in extracted_fields.items():
                # Check if the field exists in the original schema
                if field in original_schema:
                    valid_fields[field] = value
                else:
                    logger.warning(f"Field {field} does not exist in the schema and will be excluded.")

            # Log the valid extracted fields for debugging
            logger.debug(f"Valid extracted fields: {valid_fields}")

            # Return only the valid changes directly under the schema name
            return valid_fields, response.usage.to_dict()

        except json.JSONDecodeError:
            logger.error(f"GPT response was not valid JSON: {raw}")
            return {}, {}

    except Exception as e:
        logger.error(f"Error processing schema update for {schema_name}: {e}")
        return {}, {}


async def gpt_response(message_history, user_message, context=None, goal=None, tone_instructions=None, schema_list=None, scope=None):
    try:
        # Initialize dictionaries to store schema changes and token usage
        schema_changes = {}
        token_usage = {}

        # If a schema_list is provided, process each schema in the list
        if schema_list and isinstance(schema_list, list):
            if "all" in scope or len(scope) == 0:
                for schema in schema_list:
                    schema_name = schema.get("name")  # Assuming each schema has a 'name' field
                    schema_data = schema.get("data", {})  # Assuming schema data is inside 'data'

                    # Dynamically process each schema type by calling gpt_schema_update
                    changes, schema_token_usage = await gpt_schema_update(
                        schema_name,
                        schema_data,  # Pass schema data for processing
                        user_message,
                        message_history,
                    )

                    # Store changes and token usage for each schema, no need to wrap them in schema name again
                    schema_changes[schema_name] = changes
                    token_usage[schema_name] = schema_token_usage
            else:
                # If scope is not "all", only process the schemas specified in the scope
                for schema_name in scope:
                    schema = next((s for s in schema_list if s.get("name") == schema_name), None)
                    if schema:
                        schema_data = schema.get("data", {})
                        changes, schema_token_usage = await gpt_schema_update(
                            schema_name,
                            schema_data,
                            user_message,
                            message_history,
                        )
                        schema_changes[schema_name] = changes
                        token_usage[schema_name] = schema_token_usage

        # Construct the context note from a list of strings (context is a list of strings)
        if context and isinstance(context, list):
            context_str = "\n\n".join(context)
            context_note = (
                f"\n\nNote: The following relevant information has been supplied as context. "
                f"Use this to better understand the conversation:\n"
                f"{context_str}"
            )
        else:
            context_note = (
                "\n\nNote: No additional context was supplied. "
                "If the question is unrelated to the topic, politely guide the conversation back on track."
            )

        # Construct final context message by appending all message history and context note
        context_message = "".join([f"{msg['role']}: {msg['message']}\n" for msg in message_history])
        context_message += context_note

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": (
                    f"{tone_instructions or 'Provide clear, professional, and helpful responses in a conversational tone. Ensure accuracy while keeping interactions natural and engaging.'}\n\n"
                    f"The goal of this conversation is: {goal}\n\n"
                    "**Guidelines for Handling Conversations:**\n"
                    "- **conversation_over** → Use this only if the user clearly states they have no further questions.\n"
                    "- **human_intervention** → Escalate only if the user asks about scheduling, availability, or if no clear answer is found in the provided context.\n"
                    "- **continue_conversation** → If the topic allows for further discussion, offer additional insights or ask if the user would like more details.\n"
                    "- **out_of_scope** → If the user's question is unrelated, acknowledge it politely and redirect the conversation back to relevant topics.\n\n"
                )
            },
            {"role": "user", "content": context_message}],
            response_format=Sentiment,
            max_tokens=16384
        )

        parsed_sentiment = response.choices[0].message.parsed
        if "I'm not sure" in parsed_sentiment.response or "I can't help with that" in parsed_sentiment.response:
            parsed_sentiment.conversation_status = "out_of_scope"

        return {
            "response": parsed_sentiment.response,
            "conversation_status": parsed_sentiment.conversation_status,  # Tracks 'conversation_over', 'human_intervention', 'continue_conversation', 'out_of_scope'
            "changes": schema_changes,  # Return schema changes from all processed schemas
            # "token_usage": token_usage  # Return token usage for all schemas
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
        data = await request.json

        id = data.get("id")
        message_history = data.get("message_history")
        tone = data.get("tone", None)
        goal = data.get("goal", None)
        scope = data.get("schema_scope", [])
        print(f"Received scope: {scope}")

        if not all([id, message_history]):
            return jsonify({"error": "Missing required fields"}), 400

        newest_message = message_history[-1]["message"]
        context = context_store.get(id)
        context_list = context.get("context", None)
        schema_list = context.get("schema_context", None)

        if not context:
            logger.info(f"No context found for id {id}. Using GPT alone.")
            return jsonify({"error": "No context found for the given id"}), 404

        logger.info("Using supplied context for GPT response.")
        gpt_response_data = await gpt_response(message_history, newest_message, context=context_list, goal=goal, tone_instructions=tone, schema_list=schema_list, scope=scope)

        # Handle response
        conversation_status = gpt_response_data["conversation_status"]
        response = gpt_response_data

        if conversation_status == 'human_intervention':
            logger.info("Human intervention required")
            return jsonify({"response": "Human intervention required"}), 200
        elif conversation_status == 'conversation_over':
            logger.info("Conversation complete")
            return jsonify({"response": "Conversation complete"}), 200
        elif conversation_status in ('continue_conversation', 'out_of_scope'):
            logger.info("Continue conversation")
            return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 400

# For deployment with Modal
@modal_app.function(
    image=image,
    secrets=[Secret.from_name("lodasoft-app-secrets")]
)
@asgi_app()
def quart_asgi_app():
    return quart_app

# Local entrypoint for running the app
@modal_app.local_entrypoint()
def serve():
    quart_app.run()
