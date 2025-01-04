import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import AsyncOpenAI
import time

class Config:
    PINECONE_API_KEY = None
    PINECONE_INDEX_NAME = None
    OPENAI_API_KEY = None
    UNSTRUCTURED_API_KEY = None
    UNSTRUCTURED_API_URL = None
    LOCAL_FILE_INPUT_DIR = None
    LOCAL_FILE_OUTPUT_DIR = None
    INPUT_DIR = None
    OUTPUT_DIR = None
    pc = None  # Pinecone client
    aclient = None  # OpenAI client

    @staticmethod
    def initialize():
        # Fetch environment variables and set them
        Config.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        Config.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        Config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        Config.UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
        Config.UNSTRUCTURED_API_URL = os.getenv("UNSTRUCTURED_API_URL")
        Config.LOCAL_FILE_INPUT_DIR = os.getenv("LOCAL_FILE_INPUT_DIR")
        Config.LOCAL_FILE_OUTPUT_DIR = os.getenv("LOCAL_FILE_OUTPUT_DIR")

        # Set up directories
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        Config.INPUT_DIR = os.path.join(BASE_DIR, Config.LOCAL_FILE_INPUT_DIR)
        Config.OUTPUT_DIR = os.path.join(BASE_DIR, Config.LOCAL_FILE_OUTPUT_DIR)
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Initialize Pinecone client
        Config.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Initialize OpenAI client
        Config.aclient = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

        print("Config initialized successfully!")
