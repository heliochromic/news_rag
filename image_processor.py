import os
import base64
from dotenv import load_dotenv
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

load_dotenv()

mm_llm = OpenAIMultiModal(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def caption_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        base64str = base64.b64encode(f.read()).decode("utf-8")

    print(f"image caption for {image_path} was created")

    image_document = ImageDocument(image=base64str, image_mimetype="image")

    return mm_llm.complete(
        prompt="Describe shortly and clearly what is in this image. If there are metrics or charts, explain them.",
        image_documents=[image_document],
        max_tokens=150,
    ).text
