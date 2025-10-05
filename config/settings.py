import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ClinicalTrials.gov API base URL
CLINICAL_TRIALS_BASE_URL = os.getenv("CLINICAL_TRIALS_BASE_URL")

# Project-wide configurations
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
    os.makedirs(path, exist_ok=True)
