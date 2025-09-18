import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

document = TextLoader("g:/SANDBOX/backend-as-service/RAG/product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunk = text_splitter.split_documents(document)
store = Chroma.from_documents(chunk, embeddings)
retriever = store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a helpful assistant for answering questions. Use the provided context to respond. 
        If the answer in unclear respond with the message I don't know. Answer the question in three consice lines.
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)
qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain) 

history_message = InMemoryChatMessageHistory()

chat_with_message_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_message,
    input_messages_key="input",
    history_messages_key="chat_history"
)

print("Chain with document")
question = input("Enter your question: ")

if question:
    response = chat_with_message_history.invoke({"input": question}, {'configurable': {'session_id': "abc123"}})
    print(response['answer'])

