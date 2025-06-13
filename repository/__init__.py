import boto3
import os
import logging
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.environ.get("DYNAMODB_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("DYNAMODB_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION_NAME")
)

boto_client = session.resource('dynamodb')

def create_dynamo_table(table_name):
    try:
        table = boto_client.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'id', 'KeyType': 'HASH'},  # Partition key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'id', 'AttributeType': 'S'},
            ],
            BillingMode='PAY_PER_REQUEST'  # No need to specify RCU/WCU
        )
        table.wait_until_exists()  # Wait until the table is ready
        logging.info(f"Table {table_name} created successfully.")
        return table
    except Exception as e:
        logging.error(f"Failed to create table {table_name}: {e}")
        raise

def get_dynamo_table(table_name):
    try:
        # Try to get the table
        table = boto_client.Table(table_name)
        table.load()  # Check if the table exists
        logging.info(f"Connected to DynamoDB table: {table_name}")
        return table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.warning(f"Table {table_name} not found. Creating table.")
            return create_dynamo_table(table_name)
        else:
            logging.error(f"Failed to connect to table {table_name}: {e}")
            raise

def delete_dynamo_table(table_name):
    try:
        table = boto_client.Table(table_name)
        table.delete()
        logging.info(f"Table {table_name} deleted successfully.")
    except ClientError as e:
        logging.error(f"Failed to delete table {table_name}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while deleting table {table_name}: {e}")
        raise

# delete_dynamo_table("message_context")
