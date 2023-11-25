"""Run the program."""

import aws
import gpt

if __name__ == "__main__":
    image_file_name: str = "ad_4.jpg"
    components = [
        "main_object",
        "other_objects",
        "style",
    ]
    vision_model = "GPT-4 Vision"

    hashed_image, hashed_image_name = aws.hash_and_scale_image(
        mode="path", image_name=image_file_name
    )
    if not aws.find_image(hashed_image_name):
        image_url = aws.upload_image(mode="path", image_name=hashed_image_name)
    else:
        image_url = f"https://{aws.BUCKET}.s3.amazonaws.com/{hashed_image_name}"
    objects = gpt.detect_labels(image_url, components)
    print(objects)
