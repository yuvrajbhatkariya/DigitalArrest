import requests

api_key = "8e8f10f563b64dc98e35e6858477657c.83O28OlhQKn0s36iSehOoJyW"

response = requests.post(
    "https://api.ollama.ai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

print(response.status_code)
print(response.text)