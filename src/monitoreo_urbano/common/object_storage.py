from __future__ import annotations

from minio import Minio

from monitoreo_urbano.common.config import settings

_bucket_ready = False


def get_minio_client() -> Minio:
    return Minio(
        settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_secure,
    )


def get_clips_client() -> Minio:
    global _bucket_ready
    client = get_minio_client()
    if not _bucket_ready:
        if not client.bucket_exists(settings.s3_bucket_clips):
            client.make_bucket(settings.s3_bucket_clips)
        _bucket_ready = True
    return client
