from __future__ import annotations

import io
import urllib.request
import uuid
from datetime import timedelta

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.object_storage import get_clips_client


def test_bucket_exists_after_get_clips_client():
    client = get_clips_client()
    assert client.bucket_exists(settings.s3_bucket_clips)


def test_upload_and_download_roundtrip():
    client = get_clips_client()
    object_key = f"test/{uuid.uuid4().hex}.bin"
    content = b"contenido de prueba"

    client.put_object(settings.s3_bucket_clips, object_key, io.BytesIO(content), length=len(content))

    response = client.get_object(settings.s3_bucket_clips, object_key)
    try:
        assert response.read() == content
    finally:
        response.close()
        response.release_conn()

    client.remove_object(settings.s3_bucket_clips, object_key)


def test_presigned_url_serves_the_object():
    client = get_clips_client()
    object_key = f"test/{uuid.uuid4().hex}.bin"
    content = b"otro contenido de prueba"
    client.put_object(settings.s3_bucket_clips, object_key, io.BytesIO(content), length=len(content))

    url = client.presigned_get_object(settings.s3_bucket_clips, object_key, expires=timedelta(minutes=5))
    with urllib.request.urlopen(url) as resp:
        assert resp.read() == content

    client.remove_object(settings.s3_bucket_clips, object_key)
