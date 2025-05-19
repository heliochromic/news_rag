import os
import streamlit as st
from vector_store import get_existing_index
from chat_engine import RAGEngine
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="AI News Chat", page_icon="🤖", layout="wide")


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_engine" not in st.session_state:
        db_path = os.path.abspath("data/chroma_db")

        if not os.path.exists(db_path):
            st.error(f"Database directory not found at: {db_path}")
            st.stop()

        try:
            st.session_state.chat_engine = RAGEngine(db_path=db_path)
        except Exception as e:
            st.error(f"Failed to initialize the chat engine: {str(e)}")
            st.stop()


def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    st.markdown(message["sources"])


def display_image(url: str, caption: str = None, score: float = None):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        caption_text = f"{caption}\n(Relevance: {score:.2f})" if score else caption
        st.image(image, caption=caption_text, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {url}")


def format_sources_with_images(sources: list) -> tuple[str, list]:
    if not sources:
        return "No sources available", []

    formatted = []
    relevant_images = []

    for source in sources:
        source_text = f"### [{source['title']}]({source['url']})\n"
        source_text += f"**Date**: {source['date']}\n"
        source_text += f"**Relevance Score**: {source['score']:.2f}\n"

        if source["images"]:
            source_text += "\n**Related Images**:\n"
            for img in source["images"]:
                source_text += f"- [{img['caption']}]({img['url']})\n"
                img_data = {
                    "url": img["url"],
                    "caption": img["caption"],
                    "score": img["score"],
                    "source_title": source["title"],
                    "source_date": source["date"],
                }
                relevant_images.append(img_data)

        source_text += f"\n**Excerpt**:\n{source['snippet']}\n\n---\n"
        formatted.append(source_text)

    relevant_images.sort(key=lambda x: x["score"], reverse=True)
    return "\n".join(formatted), relevant_images


def main():
    st.title("AI News ChatBot 🤖")
    st.markdown(
        """
        Welcome! Ask me anything about AI news from The Batch newsletter.
        I'll provide answers with relevant sources and context from the articles.
        """
    )

    initialize_session_state()
    display_chat_messages()

    if prompt := st.chat_input("What would you like to know about AI?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.chat_engine.query(
                        prompt, chat_history=st.session_state.messages
                    )

                    st.markdown(response)

                    if sources:
                        formatted_sources, relevant_images = format_sources_with_images(
                            sources
                        )

                        if relevant_images:
                            st.write("### Related Visualizations")
                            cols = st.columns(min(2, len(relevant_images)))
                            for idx, img in enumerate(relevant_images):
                                with cols[idx % 2]:
                                    caption = f"{img['caption']}\nFrom: {img['source_title']} ({img['source_date']})"
                                    display_image(img["url"], caption, img["score"])

                        with st.expander("View Sources"):
                            st.markdown(formatted_sources)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "sources": formatted_sources,
                                "images": relevant_images,
                            }
                        )
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "I apologize, but I encountered an error while processing your question. Please try again.",
                        }
                    )


if __name__ == "__main__":
    main()
