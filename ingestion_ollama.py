import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.graph_vectorstores.networkx import documents_to_networkx
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__== "__main__":
    print("ingesting")
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter= CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts=text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    #embeddings= OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    #embeddings= OllamaEmbeddings(
     #   model="llama3.1"
    #)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


    print("Ingesting..")
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ['INDEX_NAME'])

    print("finish")
