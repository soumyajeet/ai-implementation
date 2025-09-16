from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()
OPENAI_APY_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_APY_KEY)

def chat_template():

    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system','you are a top financial expert in India. Answer any question related to personal finance.'
            'Avoid answering anything related to stock advice, mutual fund selecting advice'),
            ('human',"{input}")
        ]
    )

    data = input("Enter your question: ")


    chain = chat_template | llm

    if chain:
        response =  chain.invoke({"input":data})
        return response.content


