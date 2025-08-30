import os
from llama_index.llms.gemini import Gemini

# export GOOGLE_API_KEY

resp = Gemini().complete("Write a poem about a magic backpack")
print(resp)