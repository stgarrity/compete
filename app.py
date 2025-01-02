
import asyncio
# import copy
# import chainlit as cl
import os

import dotenv
import openai

from langsmith import traceable
from langsmith.wrappers import wrap_openai
# from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.weaviate import WeaviateVectorStore
# import weaviate
# from weaviate.classes.init import Auth

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.2,
    "max_tokens": 500
}

# wcd_url = os.environ["WCD_URL"]
# wcd_api_key = os.environ["WCD_API_KEY"]

# weaviate_client = weaviate.connect_to_weaviate_cloud(
#     cluster_url=wcd_url,                                    
#     auth_credentials=Auth.api_key(wcd_api_key),
# )

# vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name="Tesla")
# index = VectorStoreIndex.from_vector_store(vector_store)
# query_engine = index.as_query_engine()
# retriever = index.as_retriever()

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
    history.append({"role": "system", "content": """
                    You are a helpful assistant. 
                    You are an expert on UserClouds, a privacy-aware infrastructure platform, and you help our sales team educate our customers effectively.
                    Given this transcript of an ongoing conversation, decide if someone has recently (in the last 3-4 sentences) mentioned 
                    a potential competitor to UserClouds. If so, respond with the competitor's name, otherwise respond with "no".
                    Some obvious competitors are Privacera, OneTrust, Cyera, and Skyflow.
                    To catch less obvious competitors, you should also look for questions about things like "how does UserClouds compare to X", 
                    and "what are the differences between UserClouds and Y"?
                    Only respond with a competitor's name or "no".
                    """})
    
    history.append({"role": "user", "content": transcript})
    response = await client.chat.completions.create(messages=history, **model_kwargs)
    
    print("***")
    print(last_paragraph)
    print(response.choices[0].message.content)
    print("***")

    # If competitor detected, pause for keystroke
    response_text = response.choices[0].message.content.lower()
    if response_text != "no":
        print("\nCompetitor detected! Press Enter to continue...")
        input()


if __name__ == "__main__":
    asyncio.run(main())
    
# @traceable
# @cl.on_message
# async def on_message(message: cl.Message):
#     # Maintain an array of messages in the user session
#     message_history = cl.user_session.get("message_history", [])
#     message_history.append({"role": "user", "content": message.content})

#     response_message = cl.Message(content="")
#     await response_message.send()

#     rag_history = copy.deepcopy(message_history)
#     rag_history.append({"role": "system", "content": "Your only job is to identify if you need extra information from the Tesla Cyber Truck's Owners Manual to answer the last message in this thread. Respond with only one word, yes or no."})
    
#     rag = await client.chat.completions.create(messages=rag_history, **model_kwargs)
#     if rag.choices[0].message.content.lower() == "yes":
#         print("retrieving data")
#         chunks = retriever.retrieve(message.content)
        
#         context = ""
#         for chunk in chunks:
#             context += chunk.text

#         message_history[len(message_history)-1]["content"] += context

#     # Pass in the full message history for each request
#     stream = await client.chat.completions.create(messages=message_history, 
#                                                 stream=True, **model_kwargs)
#     async for part in stream:
#         if token := part.choices[0].delta.content or "":
#             await response_message.stream_token(token)

#     await response_message.update()

#     # Record the AI's response in the history
#     message_history.append({"role": "assistant", "content": response_message.content})
#     cl.user_session.set("message_history", message_history)