# import google.generativeai as genai
# import os

# genai.configure(api_key=os.environ["API_KEY"])

# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Write a story about a magic backpack.")
# print(response.text)

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Who do you think should be the coach for SF giants",
)

print(response.text)

