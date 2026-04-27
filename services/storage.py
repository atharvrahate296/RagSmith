"""
RAGSmith – File storage abstraction
Supports local disk (dev) and AWS S3 (production).
Controlled by STORAGE_BACKEND env var.

Local → data/uploads/{project_id}/{filename}
S3    → s3://{S3_BUCKET_NAME}/uploads/{project_id}/{filename}
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger("ragsmith.storage")


# ── Public API ────────────────────────────────────────────────────────────────

def save_upload(file_bytes: bytes, project_id: int, filename: str) -> str:
    """
    Persist an uploaded file.
    Returns a string key/path that can be passed to load_upload() later.
    """
    from config import get_settings
    cfg = get_settings()
    if cfg.storage_backend == "s3":
        return _s3_save(file_bytes, project_id, filename, cfg)
    return _local_save(file_bytes, project_id, filename, cfg)


def load_upload(file_key: str) -> bytes:
    """Load a previously saved file by its key/path."""
    from config import get_settings
    cfg = get_settings()
    if cfg.storage_backend == "s3":
        return _s3_load(file_key, cfg)
    return _local_load(file_key)


def delete_upload(file_key: str) -> None:
    """Delete a previously saved file."""
    from config import get_settings
    cfg = get_settings()
    if cfg.storage_backend == "s3":
        _s3_delete(file_key, cfg)
    else:
        _local_delete(file_key)


# ── Local disk ────────────────────────────────────────────────────────────────

def _local_save(file_bytes: bytes, project_id: int, filename: str, cfg) -> str:
    upload_dir = Path(cfg.local_upload_dir) / str(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(file_bytes)
    logger.info("Saved locally: %s (%d bytes)", file_path, len(file_bytes))
    return str(file_path)


def _local_load(file_key: str) -> bytes:
    path = Path(file_key)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_key}")
    return path.read_bytes()


def _local_delete(file_key: str) -> None:
    path = Path(file_key)
    if path.exists():
        path.unlink()
        logger.info("Deleted local file: %s", file_key)


# ── AWS S3 ────────────────────────────────────────────────────────────────────

def _s3_client(cfg):
    """Build a boto3 S3 client. On EC2 with an IAM role, credentials are optional."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 not installed. Run: pip install boto3") from exc

    kwargs = {"region_name": cfg.aws_region}
    # Only pass explicit credentials if provided — EC2 IAM roles work without them
    if cfg.aws_access_key_id and cfg.aws_secret_access_key:
        kwargs["aws_access_key_id"] = cfg.aws_access_key_id
        kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key

    return boto3.client("s3", **kwargs)


def _s3_key(project_id: int, filename: str) -> str:
    return f"uploads/{project_id}/{filename}"


def _s3_save(file_bytes: bytes, project_id: int, filename: str, cfg) -> str:
    if not cfg.s3_bucket_name:
        raise RuntimeError("S3_BUCKET_NAME is not configured.")

    s3 = _s3_client(cfg)
    key = _s3_key(project_id, filename)
    s3.put_object(
        Bucket=cfg.s3_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType=_content_type(filename),
    )
    logger.info("Uploaded to S3: s3://%s/%s (%d bytes)", cfg.s3_bucket_name, key, len(file_bytes))
    # Return the S3 URI as the key so load_upload / delete_upload can use it
    return f"s3://{cfg.s3_bucket_name}/{key}"


def _s3_load(file_key: str, cfg) -> bytes:
    """file_key is either 's3://bucket/key' or 'bucket/key'."""
    s3 = _s3_client(cfg)
    bucket, key = _parse_s3_uri(file_key, cfg.s3_bucket_name)
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _s3_delete(file_key: str, cfg) -> None:
    s3 = _s3_client(cfg)
    bucket, key = _parse_s3_uri(file_key, cfg.s3_bucket_name)
    s3.delete_object(Bucket=bucket, Key=key)
    logger.info("Deleted from S3: s3://%s/%s", bucket, key)


def _parse_s3_uri(uri: str, default_bucket: str):
    """Parse 's3://bucket/key' or fall back to 'bucket/key'."""
    if uri.startswith("s3://"):
        uri = uri[5:]
    parts = uri.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return default_bucket, uri


def _content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    types = {
        ".pdf":  "application/pdf",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv":  "text/csv",
        ".rst":  "text/plain",
    }
    return types.get(ext, "application/octet-stream")
