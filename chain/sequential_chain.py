from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)

# Prompt Template
title_prompt = PromptTemplate(
    input_variables=["topic"], 
    template="""
    As an expert public speaker, you need to curate a impactful speech on the following topic: {topic}.
    Answer with a tile as well.
    """)


speech_prompt = PromptTemplate(
    input_variables=["title"], 
    template="""
    You need to write powerful speech in 350 words for the following title: {title}""")

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: ("",title)[1])
secnd_chain = speech_prompt | llm
final_chain = first_chain | secnd_chain 


st.title("Curate a speech")
topic = st.text_input("Enter the topic")

if topic:
    res = final_chain.invoke({"topic": topic})
    st.write(res.content)