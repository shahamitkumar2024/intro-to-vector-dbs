
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

if __name__ == "__main__" :
    print("Retrieving..")

    #embeddings= OpenAIEmbeddings()
    #embeddings = OllamaEmbeddings(
    #    model="llama3.1"
    #)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


    #llm= ChatOpenAI()
    llm = ChatOllama(model="llama3.1")


    query = ("What is pine cone in machine learning in 10 words only")
    #chain = PromptTemplate.from_template(template=query) | llm
    #result= chain.invoke( input= {} )
    #print(result.content)

    vectorStore= PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(

        retriever=vectorStore.as_retriever(),combine_docs_chain=combine_docs_chain
    )

    result=retrival_chain.invoke({"input":query})

    print(result)