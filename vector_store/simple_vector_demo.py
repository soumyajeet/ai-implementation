import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import numpy as np


load_dotenv()
print(os.getcwd())
OPENAI_APY_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_APY_KEY)


document = TextLoader("g:/SANDBOX/backend-as-service/vector_store/support_doc.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)



chunks = text_splitter.split_documents(document)
store = Chroma.from_documents(chunks,llm)
retriever = store.as_retriever()

text = input("Input your query: ")
# embedded_data = llm.embed_query(text)
docs = retriever.invoke(text)

# docs = store.similarity_search_by_vector(embedded_data)


for doc in docs:
    print(doc.page_content)