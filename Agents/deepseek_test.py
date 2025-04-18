import requests
import json
import os
import socket
import ssl

# Temporary SSL context configuration
ctx = ssl.create_default_context()
ctx.set_ciphers('DEFAULT@SECLEVEL=1')  # Relax security level for testing

# Verify network connectivity
try:
    with socket.create_connection(("openrouter.ai", 443), timeout=5) as sock:
        with ctx.wrap_socket(sock, server_hostname="openrouter.ai") as ssock:
            print("SSL Connection verified:", ssock.version())
except Exception as e:
    print(f"Connection test failed: {e}")
    exit(1)

# Explicitly check for API key
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    print("1. Get your key from https://openrouter.ai/keys")
    print("2. Set it temporarily with:")
    exit(1)

headers = {
    "Authorization": f"Bearer {API_KEY.strip()}",  # Remove any whitespace
    "HTTP-Referer": "https://localhost:8080",  # Must be valid URL format
    "X-Title": "Python API Test",  # Max 32 chars
    "Content-Type": "application/json"
}

def ask_question(question):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"API Error: {str(e)}"

# Interactive loop
print("DeepSeek Chat Interface (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    answer = ask_question(user_input)
    print("\nAssistant:", answer)

print("\nSession ended. Goodbye!")