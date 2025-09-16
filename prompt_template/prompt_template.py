from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
prompt_templat = PromptTemplate(input_variables=["question", "country", "language"], template="""
    As an expert globetrotter tell me the capital and some interesting facts about {question}.
    Answer in {language} language and in {country} country.
    Avoid answering, if the country does not exist or it is a functional country, say "I don't know".""")


question = st.text_input("Enter your question")
language = st.text_input("Enter your language")
country = st.text_input("Enter your country")

if st.button("Submit"):
    response = llm.invoke(prompt_templat.format(question=question, language=language, country=country))
    st.write(response.content)