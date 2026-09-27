from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=1100,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

chain = chat_template | model

result = chain.invoke({'domain': 'cricket', 'topic': 'Dusra'})

print(result.content)