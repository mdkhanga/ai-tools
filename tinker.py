from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="llama2")

answer = llm.invoke("how can langsmith help with testing?")

print(answer)