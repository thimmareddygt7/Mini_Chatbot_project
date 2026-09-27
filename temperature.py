from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=1.0,  # capped from 1.5 — HF-hosted models reject values above 1.0
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)