import os
from langchain_openai import ChatOpenAI
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import wikipedia
import duckduckgo_search
from dotenv import load_dotenv 



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)


prompt = hub.pull("hwchase17/react")
tools = load_tools(["wikipedia","ddg-search"])
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

task = input("Assign a task")

response = agent_executor.invoke({"input": task})

if response:
    print(response["output"])