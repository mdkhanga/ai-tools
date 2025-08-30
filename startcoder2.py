import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "starcoder2:7b",
    "prompt": "Write a Java factorial class.",
    "stream": False  # set to False if you want a single JSON blob
}

response = requests.post(url, json=payload, stream=True)

print(response.json()["response"])