# Document Retrieval System

A web scraping and document retrieval system that extracts content from websites, processes it through vector embeddings, and provides an interactive search interface.

## How it works

The system scrapes web content and organizes it into chapters based on document structure. Each chapter gets its own chunk with attached images, making retrieval more contextual. Generated image captions help build better embeddings for more accurate search results.

## Running the project

### Step 0: Install project and dependencies

```bash
git clone https://github.com/heliochromic/news_rag
cd news_rag
```
Then create a virtual environment and install dependencies

```bash
pip install -r requirements.txt
```

Also add creds to the `.env` file 

### Step 1: Scrape and process data
```bash
python scrapper.py
```

This will scrape websites, generate image captions, create chunks, and load everything into ChromaDB using L2 similarity scoring.

### Step 2: Start the search interface
```bash
streamlit run main.py
```

Open the Streamlit app to search through the processed content.

## Key features

Documents get divided into chapters rather than arbitrary text blocks, preserving content structure and meaning.

Each chunk keeps its associated images, maintaining the connection between visual and textual content. We used generation of caption for each image to give model a knowledge about what is happening on it

From the top 5 matching chunks, only the 3 most recent are returned to avoid outdated information.

Instead of separate stores for text and images, everything stays together to prevent irrelevant image matches. We paste links to metadata to keep image information deateched to text chunks

## Technical stack

**Core model** - GPT-4o-mini for text processing and image caption generation

**Framework** - LlamaIndex for document processing and retrieval workflows

**Vector database** - ChromaDB for storing and searching embeddings

**Embeddings** - BAAI/bge-small-en-v1.5 via HuggingFace
```python
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

## Technical decisions

The multimodal approach was skipped intentionally. Since each chapter already has its own attached images, splitting into separate vector stores would risk returning unrelated pictures. The current setup ensures images always match their corresponding text context.

## Demo

![demo](demo.gif)