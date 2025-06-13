import logging
from quart import jsonify
from config import Config
from pydantic import BaseModel
from repository import get_dynamo_table
from botocore.exceptions import ClientError
from openai import OpenAIError, RateLimitError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageContextLodasoft:
    def __init__(self):
        self.table_name = "message_context_lodasoft"
        self.table = get_dynamo_table(self.table_name)

    def get(self, id):
        try:
            response = self.table.get_item(Key={"id": id})
            if "Item" in response:
                item = response["Item"]
                logging.info(f"Message context for id {id} retrieved successfully.")
                return item.get("context", [])
            else:
                logging.warning(f"No message context found for id {id}.")
                return []
        except ClientError as e:
            logging.error(f"Failed to retrieve message context for id {id}: {e}")
            raise

    def get_all(self):
        try:
            response = self.table.scan()
            items = response.get("Items", [])
            contexts = {item["id"]: item.get("context", []) for item in items}
            logging.info("All message contexts retrieved successfully.")
            return contexts
        except ClientError as e:
            logging.error(f"Failed to retrieve all message contexts: {e}")
            raise

    def delete(self, id):
        try:
            self.table.delete_item(Key={"id": id})
            logging.info(f"Message context for id {id} deleted successfully.")
        except ClientError as e:
            logging.error(f"Failed to delete message context for id {id}: {e}")
            raise

    async def update_message_context(self, id, context):
        try:
            # context is a list of strings or list of context objects
            self.table.update_item(
                Key={"id": id},
                UpdateExpression="SET context = :context",
                ExpressionAttributeValues={
                    ":context": context
                },
                ReturnValues="UPDATED_NEW"
            )
            logging.info(f"Message context for id {id} saved successfully.")
        except ClientError as e:
            logging.error(f"Failed to save message context for id {id}: {e}")
            raise

    def delete_id(self, id):
        try:
            self.table.delete_item(Key={"id": id})
            logging.info(f"Message context for id {id} deleted successfully.")
            return jsonify({"message": "Message context deleted successfully."}), 200
        except ClientError as e:
            logging.error(f"Failed to delete message context for id {id}: {e}")
            return jsonify({"error": "Failed to delete message context for id."}), 500

