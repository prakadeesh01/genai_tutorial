from llama_index.core import SimpleDirectoryReader
import sys
import tempfile
import os
from exception import CustomException
from logger import logging


def load_data(uploaded_file):
    """
    Load PDF documents from an uploaded file.

    Parameters:
    - uploaded_file (UploadedFile or file-like object): PDF uploaded via Streamlit or similar.

    Returns:
    - A list of loaded PDF documents.
    """
    try:
        logging.info("Data loading started...")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Define temp file path
            temp_file_path = os.path.join(tmpdir, uploaded_file.name)

            # Write uploaded file bytes into the temp file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load data from that temp dir
            loader = SimpleDirectoryReader(tmpdir)
            documents = loader.load_data()

        logging.info("Data loading completed...")
        return documents

    except Exception as e:
        logging.info("Exception in loading data...")
        raise CustomException(e, sys)


    