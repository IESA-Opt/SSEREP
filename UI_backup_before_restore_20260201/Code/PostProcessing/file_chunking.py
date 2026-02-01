"""
File chunking utilities for large CSV files.
Reads chunked CSV files back as a single DataFrame.
"""

import os
import pandas as pd
import json
from typing import Optional


def get_metadata_path(original_filepath: str) -> str:
    """Get path for chunk metadata file."""
    directory = os.path.dirname(original_filepath)
    filename = os.path.splitext(os.path.basename(original_filepath))[0]
    return os.path.join(directory, f"{filename}_chunk_metadata.json")


def load_chunk_metadata(metadata_path: str) -> dict:
    """Load chunk metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def read_chunked_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file that may be chunked into multiple parts.
    
    If the file exists directly, reads it normally.
    If chunks exist (indicated by metadata file), reads and concatenates all chunks.
    
    Args:
        filepath: Path to the original CSV file
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with all data
    """
    # First check if the file exists directly
    if os.path.exists(filepath):
        return pd.read_csv(filepath, **kwargs)
    
    # Check for chunked version
    metadata_path = get_metadata_path(filepath)
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"File not found: {filepath} (and no chunk metadata found)")
    
    # Load metadata and read chunks
    metadata = load_chunk_metadata(metadata_path)
    directory = os.path.dirname(filepath)
    
    chunks = []
    for chunk_file in metadata['chunk_files']:
        chunk_path = os.path.join(directory, chunk_file)
        if os.path.exists(chunk_path):
            chunk_df = pd.read_csv(chunk_path, **kwargs)
            chunks.append(chunk_df)
    
    if not chunks:
        raise FileNotFoundError(f"No chunk files found for: {filepath}")
    
    return pd.concat(chunks, ignore_index=True)
