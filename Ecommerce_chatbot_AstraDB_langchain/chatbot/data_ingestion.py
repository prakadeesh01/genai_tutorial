from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from chatbot.data_preperation import dataconveter

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRADB_ENDPOINT = os.getenv("ASTRADB_ENDPOINT")
ASTRADB_TOKEN = os.getenv("ASTRADB_TOKEN")
ASTRADB_KEYSPACE = os.getenv("ASTRADB_KEYSPACE")

embedding = GoogleGenerativeAIEmbeddings(
    model = "models/embedding-001",
    google_api_key = GOOGLE_API_KEY
)

def ingestdata(status):
    vstore = AstraDBVectorStore(
            embedding = embedding,
            collection_name = "ecommerce_chatbot_tutorial_collection",
            api_endpoint = ASTRADB_ENDPOINT,
            token = ASTRADB_TOKEN,
            namespace = ASTRADB_KEYSPACE,
        )
    
    storage=status
    
    if storage==None:
        docs=dataconveter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__=='__main__':
    vstore,inserted_ids=ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
            print(f"* {res.page_content} [{res.metadata}]")
            

   