"""OpenAI GPT API calls."""

import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def detect_labels(image_url: str, components: list) -> str:
    """Detect labels in an image."""
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You recognize objects and styles in images by classifing each of the components given to you. You respond with the component names and their corresponding labels. If the component does not exist, you label it as 'None'. If you can't calssify it, you label it as 'Unknown'. You don't add any additional text.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Identify the following components in this image: {components}",
                    },
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            },
        ],
        max_tokens=128,
    )
    return response.choices[0].message.content.split("\n")
