from typing import Tuple, List, Dict
from vector_store import get_existing_index
from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings
from datetime import datetime

QA_TEMPLATE = PromptTemplate(
    """You are an AI assistant helping with questions about AI and machine learning news from "The Batch" newsletter.
    Answer the question based on the provided context, prioritizing the most recent information.
    If you reference any charts, graphs, or metrics from images, mention them explicitly.
    Be concise and accurate.
    
    Context: {context_str}
    
    Question: {query_str}
    
    Answer: """
)


class RAGEngine:
    def __init__(self, db_path="data/chroma_db"):
        self.index = get_existing_index(db_path)
        if not self.index:
            raise ValueError("No index found in the specified path")
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5, text_qa_template=QA_TEMPLATE
        )

    def query(
        self, question: str, chat_history: List[dict] = None
    ) -> Tuple[str, List[Dict]]:
        response = self.query_engine.query(question)
        used_sources = []
        all_nodes = []

        for node in response.source_nodes:
            if node.score >= 0.4:
                date_str = node.metadata.get("published_date", "")
                try:
                    date = (
                        datetime.strptime(date_str, "%Y-%m-%d")
                        if date_str
                        else datetime.min
                    )
                except ValueError:
                    date = datetime.min

                all_nodes.append((date, node))

        all_nodes.sort(key=lambda x: x[0], reverse=True)
        selected_nodes = all_nodes[:3]

        response_text = str(response).lower()
        for _, node in selected_nodes:
            source_info = {
                "title": node.metadata.get("title", ""),
                "url": node.metadata.get("url", ""),
                "date": node.metadata.get("published_date", ""),
                "score": node.score,
                "snippet": (
                    node.text[:500] + "..." if len(node.text) > 500 else node.text
                ),
                "images": [],
            }

            for img in node.metadata.get("images", []):
                img_caption = img.get("caption", "").lower()
                if (
                    img_caption in response_text
                    or any(word in img_caption for word in question.lower().split())
                    or node.score >= 0.7
                ):
                    source_info["images"].append(
                        {
                            "url": img["url"],
                            "caption": img["caption"],
                            "score": node.score,
                        }
                    )

            if source_info["images"] or node.score >= 0.4:
                used_sources.append(source_info)

        used_sources.sort(key=lambda x: (bool(x["images"]), x["score"]), reverse=True)

        response_str = str(response)
        if used_sources:
            newest_date = used_sources[0]["date"]
            response_str = (
                f"Based on articles as recent as {newest_date}:\n\n{response_str}"
            )

            if any(source["images"] for source in used_sources):
                response_str += (
                    "\n\n*Related visualizations are available in the sources below.*"
                )

        return response_str, used_sources
