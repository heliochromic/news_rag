import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, Settings

from image_processor import caption_image


def setup_chromadb(db_path=None, collection_name="documents"):
    if db_path:
        os.makedirs(db_path, exist_ok=True)

    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        print(f"ChromaDB setup complete at: {db_path}")
        print(
            f"Collection '{collection_name}' contains {chroma_collection.count()} documents"
        )

        return vector_store, chroma_client, chroma_collection
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise


def get_existing_index(db_path="data/chroma_db", collection_name="documents"):
    db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        print(f"Database directory not found: {db_path}")
        return None

    try:
        chroma_client = chromadb.PersistentClient(path=db_path)

        chroma_collection = chroma_client.get_collection(collection_name)

        if chroma_collection.count() == 0:
            print(f"Collection '{collection_name}' is empty")
            return None

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        index = VectorStoreIndex.from_vector_store(vector_store)

        return index

    except ValueError:
        print(f"Collection '{collection_name}' not found")
        return None
    except Exception as e:
        print(f"Error loading index: {e}")
        return None


def data_load(scraped_results, vector_store):
    documents = []

    for result in scraped_results:
        if result["status"] != "success":
            continue

        for chapter in result["chapters"]:
            text_content = chapter["content"]
            image_captions = []
            image_descriptions = []

            for img in chapter.get("images", []):
                img["caption"] = caption_image(image_path=img["path"])

                if img.get("caption"):
                    image_captions.append(img["caption"])
                    image_descriptions.append(
                        {
                            "url": img["url"],
                            "caption": img["caption"],
                            "path": img["path"],
                        }
                    )

            full_text = text_content
            if image_captions:
                full_text += " " + " ".join(image_captions)

            metadata = {
                "title": chapter["title"],
                "url": chapter["url"],
                "published_date": chapter["published_date"],
                "article_id": chapter["article_id"],
                "images": image_descriptions,
            }

            doc = Document(text=full_text, metadata=metadata)
            documents.append(doc)

    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

    print(f"Successfully loaded {len(documents)} documents into ChromaDB")
    return index
