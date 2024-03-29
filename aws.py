"""AWS S3 image upload and retrieval."""

import io
from hashlib import sha256
from PIL import Image
import boto3

BUCKET = "tagger-imgs"


def hash_and_scale_image(mode: str, image_name: str = None, image_file=None):
    """Hash and scale image."""
    if mode == "path":
        with open(f"images/{image_name}", "rb") as f:
            image_content = f.read()
    elif mode == "file":
        image_content = image_file.read()
        image_name = image_file.name

    image = Image.open(io.BytesIO(image_content))
    image.thumbnail((1024, 1024))
    suffix = image_name.split(".")[-1]
    hashed_name = sha256(image_content).hexdigest() + "." + suffix

    if mode == "path":
        image.save(f"images/{hashed_name}")

    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")
    return image_stream.getvalue(), hashed_name


def find_image(image_name: str) -> bool:
    """Check if image exists in S3."""
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(BUCKET)
    for obj in bucket.objects.all():
        if obj.key == image_name:
            return True
    return False


def upload_image(mode: str, image_name: str = None, image_file=None) -> str:
    """Upload image from images folder to S3."""
    uploader = boto3.client(service_name="s3")
    if mode == "path":
        uploader.upload_file(
            Filename=f"images/{image_name}", Bucket=BUCKET, Key=image_name
        )
    elif mode == "file":
        uploader.put_object(Body=image_file, Bucket=BUCKET, Key=image_name)
    return f"https://{BUCKET}.s3.amazonaws.com/{image_name}"
