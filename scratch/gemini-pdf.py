import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
sample_pdf = genai.upload_file("/Users/Manoj/mjdocuments/papers-to-read/mvcc.pdf")

response = model.generate_content(["In a couple of lines explain mvcc", sample_pdf])
print(response.text)