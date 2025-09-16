from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_templat = PromptTemplate(
    input_variables=["country", "budget"], 
    template="""
    As an experience travel guide share the best places to visit when I visit the {country}.
    I have the budget of {budget} rupees.
    Avoid answering, if the country does not exist or it is a functional country, say "I don't know".""")


country = input("Enter your country: ")
budget = input("Enter the budget: ")
chain = prompt_templat | llm | StrOutputParser();

def chain_response():
    if chain:
        response = chain.invoke({"country": country, "budget":budget})
        return response
        