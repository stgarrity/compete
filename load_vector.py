import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth

load_dotenv()

wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
)

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

Settings.embed_model = embed_model

# Ingest the docs in the data folder

docs = SimpleDirectoryReader('./data').load_data()

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(docs)

# construct vector store
vector_store = WeaviateVectorStore(weaviate_client = client, index_name=os.environ["WCD_INDEX_NAME"])

# setting up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store = vector_store)

# set up the index
index = VectorStoreIndex(nodes, storage_context = storage_context)