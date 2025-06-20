from quart_cors import cors
from functools import wraps
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
modal_app = App("bonzo-app")
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

        if not id or not context or not isinstance(context, list):
            return jsonify({"error": "Missing required fields: id and context are required."}), 400

        # Save context to DynamoDB
        await context_store.update_message_context(id, context)

        logging.info(f"Context uploaded successfully for id {id}.")
        return jsonify({"message": "Context uploaded successfully.", "id": id, "context": context}), 200

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

async def gpt_schema_update(aclient, schema_type: str, original: dict, user_message: str) -> tuple[dict, dict]:
    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a data assistant. The following is the original {schema_type} schema. "
                "The user has provided a message that may contain updates to some fields in the schema.\n\n"
                "Your task is to extract ONLY the fields clearly mentioned in the message and return them as a nested object.\n"
                "- If a field is mentioned, include it in the result with its new value.\n"
                "- If a field is not mentioned, do not include it.\n"
                "- If a field is nested (inside another object), return it as a nested object (e.g. { \"solar\": { \"firstName\": \"John\" } }).\n"
                "- If the same field name appears in multiple places in the schema, nest it correctly inside its parent object. Do not use dot notation. For example: { \"solar\": { \"firstName\": \"John\" } }.\n"
                "- If the user message provides general personal information (e.g. name, phone, email) without specifying context, map it to the most general applicant-level fields first (e.g. \"loanApplicant.firstName\" if applicable).\n"
                "- Do not make assumptions. If the user message is ambiguous, only extract fields when there is a clear match. Otherwise leave them out.\n\n"
                "Return ONLY the following JSON structure:\n"
                '{ \"extracted_fields\": { \"field_name\": value, ... } }\n\n'
                "Do not include the entire updated schema. Only include the fields mentioned in the user message.\n"
                "Respond ONLY with a JSON object. Do not include ```json or any explanation."
            )
        },
        {
            "role": "user",
            "content": json.dumps({
                "original_schema": original,
                "user_message": user_message
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
        logger.info(f"Parsed schema update: {parsed}")

        # Extract token usage
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Return extracted_fields only, token_usage
        return parsed.get("extracted_fields", {}), token_usage

    except json.JSONDecodeError:
        logger.error("GPT response was not valid JSON:\n" + raw)

        # Return empty dict with token usage if available
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }

        return {}, token_usage

async def gpt_response(message_history, user_message, contexts=None, goal=None, tone_instructions=None, scope="all"):
    # aclient = client

    # lead_data, loan_application_model = await get_lead_info(lead_id, loan_id) if lead_id and loan_id else (None, None)

    try:

        # if scope == "all":
        #     lead_changes, lead_token_usage = await gpt_schema_update(aclient, "lead", lead_schema or {}, user_message)
        #     loan_changes, loan_token_usage = await gpt_schema_update(aclient, "loan", loan_schema or {}, user_message)
        # elif scope == "lead_info":
        #     lead_changes, lead_token_usage = await gpt_schema_update(aclient, "lead", lead_schema or {}, user_message)
        #     loan_changes, loan_token_usage = {}, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        # elif scope == "loan_info":
        #     lead_changes, lead_token_usage = {}, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        #     loan_changes, loan_token_usage = await gpt_schema_update(aclient, "loan", loan_schema or {}, user_message)

        change_log = ""

        if contexts and isinstance(contexts, list):
            context_str = "\n\n".join(contexts)
            context_note = (
                f"\n\nNote: The following relevant information has been supplied as context. "
                "Use this to better understand the conversation.\n"
                f"{context_str}"
            )
        else:
            context_note = (
                "\n\nNote: No additional context was supplied. "
                "If the question is unrelated to the topic, politely guide the conversation back on track."
            )

        # Construct final context message
        context_message = "".join([f"{msg['role']}: {msg['message']}\n" for msg in message_history])
        context_message += context_note

        if change_log:
            context_message += f"\n\nThe following updates were inferred from the conversation:\n{change_log}"

        response = await aclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{tone_instructions or 'Provide clear, professional, and helpful responses in a conversational tone. Ensure accuracy while keeping interactions natural and engaging.'}\n\n"

                        f"The goal of this conversation is: {goal}\n\n"

                        "**Guidelines for Handling Conversations:**\n"
                        "- **conversation_over** → Use this only if the user clearly states they have no further questions.\n"
                        "- **human_intervention** → Escalate only if the user asks about scheduling, availability, or if no clear answer is found in the provided context.\n"
                        "- **continue_conversation** → If the topic allows for further discussion, offer additional insights or ask if the user would like more details.\n"
                        "- **out_of_scope** → If the user's question is unrelated, acknowledge it politely and redirect the conversation back to relevant topics.\n\n"

                        "**Handling Out-of-Scope Questions:**\n"
                        "If a user asks something unrelated, respond in a way that maintains a natural flow:\n"
                        "👤 User: 'What's the best Italian restaurant nearby?'\n"
                        "💬 Response: 'That sounds like a great topic! While I don't have restaurant recommendations, I'd be happy to assist with [specific topic]. Let me know how I can help!'\n\n"

                        "If the user continues with off-topic questions, acknowledge their curiosity but steer the conversation back in a professional and engaging manner."
                        "DO NOT USE EMOTICONS OR EMOJIS IN YOUR RESPONSES EVER.\n\n"
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

        # total_response_tokens = token_usage.get("total_tokens", 0)
        # total_lead_tokens = lead_token_usage.get("total_tokens", 0)
        # total_loan_tokens = loan_token_usage.get("total_tokens", 0)

        return {
            "response": parsed_sentiment.response,
            "conversation_status": parsed_sentiment.conversation_status,  # Tracks 'conversation_over', 'human_intervention', 'continue_conversation', 'out_of_scope'
            "changes": {
                # "lead_schema_data": lead_changes,
                # "loan_schema_data": loan_changes
            },
            # "token_usage": {
            #     "response_tokens": total_response_tokens,
            #     "lead_schema_tokens": total_lead_tokens,
            #     "loan_schema_tokens": total_loan_tokens
            # }
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
        # context = data.get("context", [])
        tone = data.get("tone", None)
        goal = data.get("goal", None)
        scope = data.get("scope", "all")

        if not all([id, message_history]):
            return jsonify({"error": "Missing required fields"}), 400

        newest_message = message_history[-1]["message"]
        context = context_store.get(id)

        if not context:
            logger.info(f"No context found for id {id}. Using GPT alone.")
            return jsonify({"error": "No context found for the given id"}), 404

        logger.info("Using supplied context for GPT response.")
        gpt_response_data = await gpt_response(message_history, newest_message, context, goal=goal, tone_instructions=tone, scope=scope)

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
    secrets=[Secret.from_name("bonzo-app-secrets")]
)
@asgi_app()
def quart_asgi_app():
    return quart_app

# Local entrypoint for running the app
@modal_app.local_entrypoint()
def serve():
    quart_app.run()
