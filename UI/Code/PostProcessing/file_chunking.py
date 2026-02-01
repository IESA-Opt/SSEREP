"""
File chunking utilities for large CSV files to make them GitHub-compatible.
Splits files into chunks smaller than 100MB and provides functions to read them back.
"""

import os
import pandas as pd
import json
import math
from typing import List, Tuple, Optional

# Default chunk size: 95MB to stay safely under 100MB GitHub limit
DEFAULT_CHUNK_SIZE_MB = 95

# Deprecated multiplier (kept for backward compatibility in functions); we now
# prefer file-size-based estimation for more uniform chunk sizes
ROWS_PER_CHUNK_MULTIPLIER = 1.0

def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def needs_chunking(filepath: str, max_size_mb: float = 100) -> bool:
    """Check if file needs to be chunked based on size."""
    return get_file_size_mb(filepath) > max_size_mb

def create_chunk_metadata(original_file: str, chunk_files: List[str], total_rows: int) -> dict:
    """Create metadata for chunked file."""
    return {
        'original_file': os.path.basename(original_file),
        'total_chunks': len(chunk_files),
        'chunk_files': [os.path.basename(f) for f in chunk_files],
        'total_rows': total_rows,
        'created_by': 'SSDashboard file chunking system',
        'version': '1.0'
    }

def save_chunk_metadata(metadata_path: str, metadata: dict) -> None:
    """Save chunk metadata to JSON file."""
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_chunk_metadata(metadata_path: str) -> dict:
    """Load chunk metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_chunks_directory(original_filepath: str) -> str:
    """Get directory where chunks should be stored (same as original file)."""
    return os.path.dirname(original_filepath)

def get_metadata_path(original_filepath: str) -> str:
    """Get path for chunk metadata file."""
    directory = os.path.dirname(original_filepath)
    filename = os.path.splitext(os.path.basename(original_filepath))[0]
    return os.path.join(directory, f"{filename}_chunk_metadata.json")

def estimate_rows_per_chunk(filepath: str, chunk_size_mb: float) -> int:
    """Estimate how many rows should be in each chunk to target the desired size."""
    file_size_mb = get_file_size_mb(filepath)
    
    # Sample the file to estimate rows
    sample_df = pd.read_csv(filepath, nrows=1000, low_memory=False)
    total_rows_estimated = file_size_mb / (get_memory_usage_mb(sample_df) / len(sample_df))
    
    # Calculate rows per chunk
    target_rows_per_chunk = int((chunk_size_mb / file_size_mb) * total_rows_estimated)
    # Apply calibration multiplier (legacy)
    target_rows_per_chunk = int(target_rows_per_chunk * ROWS_PER_CHUNK_MULTIPLIER)
    
    # Ensure minimum chunk size
    return max(target_rows_per_chunk, 1000)

def estimate_rows_per_chunk_by_filesize(filepath: str, target_mb: float) -> int:
    """Estimate rows per chunk using file size and total row count for uniform chunks.

    Approach:
    - Count data rows quickly (excluding header)
    - Compute bytes per row from file size
    - Return rows_per_chunk ≈ target_mb / bytes_per_row
    """
    size_bytes = os.path.getsize(filepath)
    # Count rows excluding header
    with open(filepath, 'rb') as f:
        header = f.readline()
        # Count remaining lines efficiently
        rows = 0
        for block in iter(lambda: f.read(1024 * 1024), b''):
            rows += block.count(b'\n')
    rows = max(rows, 1)  # avoid div by zero
    header_len = len(header)
    data_bytes = max(size_bytes - header_len, 1)
    bytes_per_row = data_bytes / rows
    target_bytes = target_mb * 1024 * 1024
    rows_per_chunk = int(target_bytes / bytes_per_row)
    return max(rows_per_chunk, 1000)

def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Get memory usage of dataframe in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def split_csv_into_chunks(filepath: str, chunk_size_mb: float = DEFAULT_CHUNK_SIZE_MB, 
                         force: bool = False) -> Tuple[bool, List[str], str]:
    """
    Split a large CSV file into smaller chunks.
    
    Args:
        filepath: Path to the CSV file to split
        chunk_size_mb: Target size for each chunk in MB
        force: If True, overwrite existing chunks
    
    Returns:
        (success, chunk_files, metadata_path)
    """
    print(f"Checking if chunking is needed for {filepath}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False, [], ""
    
    # Check if file needs chunking
    file_size = get_file_size_mb(filepath)
    print(f"File size: {file_size:.1f} MB")
    
    if not needs_chunking(filepath, 100) and not force:
        print("File is smaller than 100MB, no chunking needed")
        return True, [filepath], ""
    
    chunks_dir = get_chunks_directory(filepath)
    metadata_path = get_metadata_path(filepath)
    
    # Check if chunks already exist and are valid
    if os.path.exists(metadata_path) and not force:
        try:
            metadata = load_chunk_metadata(metadata_path)
            chunk_paths = [os.path.join(chunks_dir, f) for f in metadata['chunk_files']]
            if all(os.path.exists(p) for p in chunk_paths):
                print(f"Chunks already exist: {len(chunk_paths)} files")
                return True, chunk_paths, metadata_path
        except Exception as e:
            print(f"Invalid existing chunks, will recreate: {e}")
    
    print(f"Splitting {filepath} into chunks of ~{chunk_size_mb}MB each...")
    
    # Chunks will be saved in the same directory as the original file
    # No need to create separate directory
    
    # Estimate optimal chunk size in rows using file-size-based approach for uniform chunks
    try:
        rows_per_chunk = estimate_rows_per_chunk_by_filesize(filepath, min(chunk_size_mb, 95.0))
        print(f"Estimated rows per chunk (file-size): {rows_per_chunk:,}")
    except Exception as e:
        print(f"Could not estimate rows from file size, falling back to sample-based: {e}")
        try:
            rows_per_chunk = estimate_rows_per_chunk(filepath, chunk_size_mb)
            print(f"Estimated rows per chunk (sample): {rows_per_chunk:,}")
        except Exception as e2:
            print(f"Could not estimate chunk size, using default: {e2}")
            rows_per_chunk = 50000  # Safe default
    
    chunk_files = []
    chunk_count = 0
    total_rows = 0

    # Limits and targets
    MAX_CHUNK_MB = 100.0
    TARGET_CHUNK_MB = min(chunk_size_mb, 95.0)  # aim around 90-95MB
    SAFETY = 0.98  # small safety margin
    
    try:
        # Read CSV in chunks and adapt chunk size to ensure < 100MB per chunk
        # Use a rolling buffer so that we only have a single final small chunk
        reader = pd.read_csv(filepath, chunksize=rows_per_chunk, low_memory=False)
        base_name = os.path.splitext(os.path.basename(filepath))[0]

        buffer_df = None  # carry leftover rows to the next iteration
        current_target_rows = rows_per_chunk

        for chunk_df in reader:
            total_rows += len(chunk_df)

            # Combine with any buffered remainder
            if buffer_df is not None and not buffer_df.empty:
                work_df = pd.concat([buffer_df, chunk_df], ignore_index=True)
            else:
                work_df = chunk_df

            start = 0
            nrows = len(work_df)

            # Write as many full-size chunks as possible from the combined data
            while (nrows - start) >= current_target_rows:
                sub_df = work_df.iloc[start:start + current_target_rows]

                # Try writing and measuring; if too big, shrink and retry
                while True:
                    chunk_count += 1
                    chunk_filename = f"{base_name}_chunk_{chunk_count:03d}.csv"
                    chunk_path = os.path.join(chunks_dir, chunk_filename)

                    sub_df.to_csv(chunk_path, index=False)
                    chunk_size = get_file_size_mb(chunk_path)

                    if chunk_size > MAX_CHUNK_MB:
                        # Remove and try with fewer rows
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass
                        factor = (TARGET_CHUNK_MB / max(chunk_size, 1e-6)) * SAFETY
                        new_rows = max(int(len(sub_df) * factor), 1000)
                        # Ensure progress
                        if new_rows >= len(sub_df):
                            new_rows = max(len(sub_df) // 2, 1000)
                        sub_df = sub_df.iloc[:new_rows]
                        # Learn downward for all subsequent chunks
                        current_target_rows = min(current_target_rows, new_rows)
                        rows_per_chunk = current_target_rows
                        continue

                    print(f"Created chunk {chunk_count}: {chunk_filename} ({chunk_size:.1f}MB, {len(sub_df):,} rows)")
                    chunk_files.append(chunk_path)
                    start += len(sub_df)
                    break

            # Buffer any remainder to combine with the next read so we don't produce
            # small chunks repeatedly. Only the very last chunk should be smaller.
            buffer_df = work_df.iloc[start:]

        # After exhausting the reader, flush the remainder (last small chunk)
        if buffer_df is not None and not buffer_df.empty:
            chunk_count += 1
            chunk_filename = f"{base_name}_chunk_{chunk_count:03d}.csv"
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            buffer_df.to_csv(chunk_path, index=False)
            chunk_size = get_file_size_mb(chunk_path)
            print(f"Created final chunk {chunk_count}: {chunk_filename} ({chunk_size:.1f}MB, {len(buffer_df):,} rows)")
            chunk_files.append(chunk_path)
        
        # Create and save metadata
        metadata = create_chunk_metadata(filepath, chunk_files, total_rows)
        save_chunk_metadata(metadata_path, metadata)

        print(f"Successfully split into {len(chunk_files)} chunks")
        print(f"Total rows: {total_rows:,}")
        print(f"Metadata saved to: {metadata_path}")

        return True, chunk_files, metadata_path
        
    except Exception as e:
        print(f"Error splitting file: {e}")
        return False, [], ""

def read_chunked_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file that may be chunked.
    
    Args:
        filepath: Path to the original CSV file (or chunk metadata)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        Combined DataFrame from all chunks, or original file if not chunked
    """
    # Check if the original file exists and is small enough
    if os.path.exists(filepath) and not needs_chunking(filepath, 100):
        return pd.read_csv(filepath, **kwargs)
    
    # Look for chunk metadata
    metadata_path = get_metadata_path(filepath)
    
    if not os.path.exists(metadata_path):
        # No chunks found, try to read original file anyway
        if os.path.exists(filepath):
            return pd.read_csv(filepath, **kwargs)
        else:
            raise FileNotFoundError(f"Neither original file nor chunks found for: {filepath}")
    
    # Load chunk metadata
    try:
        metadata = load_chunk_metadata(metadata_path)
        chunks_dir = get_chunks_directory(filepath)
        
        print(f"Reading {metadata['total_chunks']} chunks ({metadata['total_rows']:,} total rows)")
        
        # Read all chunks
        chunk_dfs = []
        for chunk_file in metadata['chunk_files']:
            chunk_path = os.path.join(chunks_dir, chunk_file)
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            
            chunk_df = pd.read_csv(chunk_path, **kwargs)
            chunk_dfs.append(chunk_df)
        
        # Combine all chunks
        combined_df = pd.concat(chunk_dfs, ignore_index=True)
        print(f"Successfully combined {len(chunk_dfs)} chunks into DataFrame with {len(combined_df):,} rows")
        
        return combined_df
        
    except Exception as e:
        print(f"Error reading chunked file: {e}")
        # Fallback to original file if it exists
        if os.path.exists(filepath):
            print("Falling back to original file")
            return pd.read_csv(filepath, **kwargs)
        else:
            raise

def chunk_exists(filepath: str) -> bool:
    """Check if chunks exist for a file."""
    metadata_path = get_metadata_path(filepath)
    if not os.path.exists(metadata_path):
        return False
    
    try:
        metadata = load_chunk_metadata(metadata_path)
        chunks_dir = get_chunks_directory(filepath)
        return all(os.path.exists(os.path.join(chunks_dir, f)) for f in metadata['chunk_files'])
    except:
        return False

def get_chunk_info(filepath: str) -> Optional[dict]:
    """Get information about chunks for a file."""
    metadata_path = get_metadata_path(filepath)
    if not os.path.exists(metadata_path):
        return None
    
    try:
        metadata = load_chunk_metadata(metadata_path)
        chunks_dir = get_chunks_directory(filepath)
        
        # Calculate total size of chunks
        total_chunk_size = 0
        for chunk_file in metadata['chunk_files']:
            chunk_path = os.path.join(chunks_dir, chunk_file)
            if os.path.exists(chunk_path):
                total_chunk_size += get_file_size_mb(chunk_path)
        
        return {
            'total_chunks': metadata['total_chunks'],
            'total_rows': metadata['total_rows'],
            'total_size_mb': total_chunk_size,
            'avg_chunk_size_mb': total_chunk_size / metadata['total_chunks'],
            'chunks_dir': chunks_dir
        }
    except:
        return None

def cleanup_chunks(filepath: str) -> bool:
    """Remove chunks for a file."""
    metadata_path = get_metadata_path(filepath)
    
    if not os.path.exists(metadata_path):
        return True  # No chunks to clean up
    
    try:
        metadata = load_chunk_metadata(metadata_path)
        chunks_dir = get_chunks_directory(filepath)
        
        # Remove each chunk file
        removed_count = 0
        for chunk_file in metadata['chunk_files']:
            chunk_path = os.path.join(chunks_dir, chunk_file)
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                removed_count += 1
        
        # Remove metadata file
        os.remove(metadata_path)
        
        print(f"Removed {removed_count} chunk files and metadata")
        return True
    except Exception as e:
        print(f"Error removing chunks: {e}")
        return False