import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()
OPENAI_APY_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_APY_KEY)


query1 = "Cat"
query2 = "Kitten"

response1 = llm.embed_query(query1)
response2 = llm.embed_query(query2)

print(response1)
print(response2)

similarity_score = np.dot(response1, response2)
print("SCORE: ",similarity_score*100)