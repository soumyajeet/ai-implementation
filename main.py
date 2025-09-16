# from chain import chain
from chathistory import chat_prompt_template, chat_history
from embeddings import embeddings_demo
import streamlit as st

    
# def chain():
#     data = chain.chain_response()
#     print(data)

def chat_template_res():
    # data = chat_prompt_template.chat_template()
    data = chat_prompt_template.chat_history()
    print(data)

    


if __name__ == "__main__":
    chat_template_res()
