import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])

uploaded_files = genai.list_files()

for file in uploaded_files:
    print(file.uri)
    
print("Done listing files")