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


# chat_template = ChatPromptTemplate.from_messages([
#     ("system",""" Answer the following questions as best you can. You have access to the following tools:

#         {tools}

#         Use the following format:

#         Question: the input question you must answer
#         Thought: you should always think about what to do
#         Action: the action to take, should be one of [{tool_names}]
#         Action Input: the input to the action
#         Observation: the result of the action
#         ... (this Thought/Action/Action Input/Observation can repeat N times)
#         Thought: I now know the final answer
#         Final Answer: the final answer to the original input question

#         Begin!

#         Question: {input}
#         Thought:{agent_scratchpad}
#     """),
#     ("human", f"{input}")
#     ])
