from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from uuid import uuid4

import os
import streamlit as st

load_dotenv()
OPENAI_APY_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_APY_KEY)
session_id = uuid4();

def chat_template():

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","you are a top financial expert in India. Answer any question related to personal finance."
            "Avoid answering anything related to stock advice, mutual fund selecting advice"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ]
    )

    chain = chat_template | llm

    history_message = InMemoryChatMessageHistory()

    chat_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history_message,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    data = st.text_input("Enter your question: ")

    if chain:
        response =  chat_with_message_history.invoke({"input":data},{'configurable': {'session_id': session_id}})
        st.write(response.content) 

chat_template()
