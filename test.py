import os
from dotenv import dotenv_values


# Load environment variables from .env file
config = dotenv_values(".env")

# Get the Hugging Face API token from environment variables

print(config)