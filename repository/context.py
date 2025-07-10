import logging
from quart import jsonify
from repository import get_dynamo_table
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageContext:
    def __init__(self):
        self.table_name = "message_context"
        self.table = get_dynamo_table(self.table_name)

    def get(self, id):
        try:
            response = self.table.get_item(Key={"id": id})
            if "Item" in response:
                item = response["Item"]
                logging.info(f"Message context for id {id} retrieved successfully.")
                # Return both context and schema_context
                return {
                    "context": item.get("context", []),
                    "schema_context": item.get("schema_context", [])
                }
            else:
                logging.warning(f"No message context found for id {id}.")
                return {"context": [], "schema_context": []}
        except ClientError as e:
            logging.error(f"Failed to retrieve message context for id {id}: {e}")
            raise

    def get_all(self):
        try:
            response = self.table.scan()
            items = response.get("Items", [])
            contexts = {
                item["id"]: {
                    "context": item.get("context", []),
                    "schema_context": item.get("schema_context", [])
                } for item in items
            }
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

    async def update_message_context(self, id, context, schema_context):
        try:
            # Save both context and schema_context in DynamoDB
            self.table.put_item(
                Item={
                    "id": id,
                    "context": context,
                    "schema_context": schema_context
                }
            )
            logging.info(f"Message context and schema_context for id {id} saved successfully.")
        except ClientError as e:
            logging.error(f"Failed to save message context and schema_context for id {id}: {e}")
            raise

    def delete_id(self, id):
        try:
            self.table.delete_item(Key={"id": id})
            logging.info(f"Message context for id {id} deleted successfully.")
            return jsonify({"message": "Message context deleted successfully."}), 200
        except ClientError as e:
            logging.error(f"Failed to delete message context for id {id}: {e}")
            return jsonify({"error": "Failed to delete message context for id."}), 500
