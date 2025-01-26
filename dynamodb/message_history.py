import logging
from quart import jsonify
from config import Config
from pydantic import BaseModel
from dynamodb import get_dynamo_table
from botocore.exceptions import ClientError
from openai import OpenAIError, RateLimitError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageHistory:
    def __init__(self):
        self.table_name = "message_history"
        self.table = get_dynamo_table(self.table_name)

    def get(self, namespace):
        try:
            response = self.table.get_item(Key={"namespace": namespace})
            if "Item" in response:
                item = response["Item"]
                logging.info(f"Message history for user {namespace} retrieved successfully.")
                return item.get("messages", [])
            else:
                logging.warning(f"No message history found for user {namespace}.")
                return []
        except ClientError as e:
            logging.error(f"Failed to retrieve message history for user {namespace}: {e}")
            raise

    def get_all(self):
        try:
            response = self.table.scan()
            items = response.get("Items", [])
            histories = {item["namespace"]: item.get("messages", []) for item in items}
            logging.info("All message histories retrieved successfully.")
            return histories
        except ClientError as e:
            logging.error(f"Failed to retrieve all message histories: {e}")
            raise

    def delete(self, namespace):
        try:
            self.table.delete_item(Key={"namespace": namespace})
            logging.info(f"Message history for user {namespace} deleted successfully.")
        except ClientError as e:
            logging.error(f"Failed to delete message history for user {namespace}: {e}")
            raise

    async def update_message_history(self, namespace, message):
        try:
            # Message format {role: "user"/"assistant", message: "message"}
            self.table.update_item(
                Key={"namespace": namespace},
                UpdateExpression="SET messages = list_append(if_not_exists(messages, :empty_list), :message)",
                ExpressionAttributeValues={
                    ":message": message,
                    ":empty_list": []
                },
                ReturnValues="UPDATED_NEW"
            )
            logging.info(f"Message history for {namespace} saved successfully.")
        except ClientError as e:
            logging.error(f"Failed to save message history for user {namespace}: {e}")
            raise

    async def get_gpt_response(self, message, namespace, res=None):
        aclient = Config().aclient

        class Sentiment(BaseModel):
            response: str
            is_conversation_over: str

        try:
            context_message = message if res is None else message + f"Use the following context: {res}"

            response = await aclient.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful sms assistant. Provide clear and concise responses to customer queries. Be professional and conversational. Answer questions based on the context provided."},
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
            await self.update_message_history(namespace, [{"role": "user", "message": message}, {"role": "assistant", "message": res_dict["response"]}])

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

    def batch_save(self, histories):
        try:
            with self.table.batch_writer() as batch:
                for namespace, messages in histories.items():
                    item = {"namespace": namespace, "messages": messages}
                    batch.put_item(Item=item)
            logging.info("Batch save operation for message histories completed successfully.")
        except ClientError as e:
            logging.error(f"Failed to save batch of message histories: {e}")
            raise
