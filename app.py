
import asyncio
import os

import dotenv
import openai

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from llama_index.core import Settings,VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth

from prompts import SYSTEM_DETECTION_PROMPT, SYSTEM_COMPETITIVE_ANALYSIS_PROMPT

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.2,
    "max_tokens": 500
}

wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
)


embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.embed_model = embed_model
vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=os.environ["WCD_INDEX_NAME"])
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
retriever = index.as_retriever()

client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))

@traceable
async def main():
    # Get list of transcript files
    transcript_dir = "../slackexporter/transcripts/"
    transcript_files = []
    for filename in os.listdir(transcript_dir):
        if os.path.isfile(os.path.join(transcript_dir, filename)):
            transcript_files.append(filename)
    print(f"Found {len(transcript_files)} transcript files")

    # Load the content of each transcript file
    transcripts = []
    for filename in transcript_files:
        # FIXME
        if "kwok" not in filename.lower():
            continue

        with open(os.path.join(transcript_dir, filename), 'r') as file:
            transcript_text = file.read()

            # Split transcript into paragraphs
            paragraphs = transcript_text.split('\n')
            paragraphs = [p for p in paragraphs if p.strip()]  # Remove empty paragraphs

            # Analyze progressively larger chunks of the transcript
            for i in range(len(paragraphs)):
                # Create chunk containing paragraphs 0 through i
                chunk = '\n\n'.join(paragraphs[:i+1])
                await analyze_chunk(chunk, paragraphs[i])


async def analyze_chunk(transcript, last_paragraph):
    history = []
    history.append({"role": "system", "content": SYSTEM_DETECTION_PROMPT})

    history.append({"role": "user", "content": transcript})
    response = await client.chat.completions.create(messages=history, **model_kwargs)

    print("***")
    print(last_paragraph)
    print(response.choices[0].message.content)
    print("***")

    # If competitor detected, pause for keystroke
    response_text = response.choices[0].message.content.lower()
    if response_text != "no":
        chunks = retriever.retrieve(response_text)
        competitor_info = "\n".join(chunk.text for chunk in chunks)
        history = []
        history.append({"role": "system", "content": SYSTEM_COMPETITIVE_ANALYSIS_PROMPT})
        history.append({"role": "user", "content": competitor_info})
        response = await client.chat.completions.create(messages=history, **model_kwargs)
        print(response.choices[0].message.content)

        print("\nCompetitor detected! Press Enter to continue...")
        input()


if __name__ == "__main__":
    asyncio.run(main())
