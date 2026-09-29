import json
from datetime import datetime
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings


class StorageError(RuntimeError):
    """Raised when the object-storage provider cannot complete an operation."""


class S3Storage:
    """Small S3-compatible storage adapter for MinIO locally and S3 in production."""

    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.storage_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.storage_endpoint_url,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            region_name=settings.storage_region,
            config=Config(
                s3={"addressing_style": "path"},
                connect_timeout=settings.storage_connect_timeout_seconds,
                read_timeout=settings.storage_read_timeout_seconds,
            ),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise StorageError(f"Unable to inspect storage bucket: {code}") from exc
        except BotoCoreError as exc:
            raise StorageError("Unable to reach object storage") from exc

        try:
            self.client.create_bucket(Bucket=self.bucket)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise StorageError(f"Unable to create storage bucket: {code}") from exc
        except BotoCoreError as exc:
            raise StorageError("Unable to create storage bucket") from exc

    def check_connection(self) -> None:
        try:
            self.client.list_buckets()
        except (BotoCoreError, ClientError) as exc:
            raise StorageError("Unable to reach object storage") from exc

    def put_bytes(self, key: str, body: bytes, content_type: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except BotoCoreError as exc:
            raise StorageError(f"Unable to store object {key}") from exc

    def put_json(self, key: str, payload: Any) -> None:
        self.put_bytes(
            key,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json",
        )

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Unable to read object {key}") from exc

    def delete_object(self, key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Unable to delete object {key}") from exc

    def list_objects(self) -> list[tuple[str, datetime]]:
        """List object keys and modification times for orphan reconciliation."""

        objects: list[tuple[str, datetime]] = []
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket):
                for item in page.get("Contents", []):
                    key = item.get("Key")
                    modified = item.get("LastModified")
                    if key and isinstance(modified, datetime):
                        objects.append((str(key), modified))
        except (BotoCoreError, ClientError) as exc:
            raise StorageError("Unable to enumerate object storage") from exc
        return objects
