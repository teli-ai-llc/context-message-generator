import os, time, logging
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load LOCAL environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # Class-level attributes
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
        # Fetch environment variables
        Config.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        Config.PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
        Config.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        Config.UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")
        Config.UNSTRUCTURED_API_URL = os.environ.get("UNSTRUCTURED_API_URL")
        Config.LOCAL_FILE_INPUT_DIR = os.environ.get("LOCAL_FILE_INPUT_DIR", "input")
        Config.LOCAL_FILE_OUTPUT_DIR = os.environ.get("LOCAL_FILE_OUTPUT_DIR", "output")

        # Validate required environment variables
        if not all([Config.PINECONE_API_KEY, Config.PINECONE_INDEX_NAME, Config.OPENAI_API_KEY]):
            raise ValueError("Missing required environment variables for Pinecone or OpenAI.")

        # Set up directories
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        Config.INPUT_DIR = os.path.join(BASE_DIR, Config.LOCAL_FILE_INPUT_DIR)
        Config.OUTPUT_DIR = os.path.join(BASE_DIR, Config.LOCAL_FILE_OUTPUT_DIR)
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Initialize Pinecone client
        try:
            Config.pc = Pinecone(api_key=Config.PINECONE_API_KEY)

            if Config.PINECONE_INDEX_NAME not in [index.name for index in Config.pc.list_indexes().indexes]:
                Config.pc.create_index(
                    name=Config.PINECONE_INDEX_NAME,
                    dimension=1024,  # Replace with dynamic dimension if needed
                    metric="cosine",  # Replace with dynamic metric if needed
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Wait until the Pinecone index is ready
                while not Config.pc.describe_index(Config.PINECONE_INDEX_NAME).status["ready"]:
                    time.sleep(1)
                logger.info(f"Index {Config.PINECONE_INDEX_NAME} created successfully!")
            else:
                logger.info(f"Index {Config.PINECONE_INDEX_NAME} already exists.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

        # Initialize OpenAI client
        try:
            Config.aclient = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

        logger.info("Config initialized successfully!")
