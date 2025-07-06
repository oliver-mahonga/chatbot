from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import requests
import os
from dotenv import load_dotenv
load_dotenv()
hf_api_token = os.getenv("HF_API_TOKEN")
if not hf_api_token:
    print("ERROR: error in hugging face API ")
app = FastAPI()
app.mount("/", StaticFiles(directory=".", html=True), name="static")


model = "distilgpt2" 
api_url = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {hf_api_token}"}

@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"response": "No text provided"}
    if not hf_api_token:
        return {"response": "Server Error: Missing API token"}

    payload = {
        "inputs": text,
        "parameters": {"max_length": 200, "num_return_sequences": 1, "temperature": 0.7}
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("API Response:", result)
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(text):
                answer = generated_text[len(text):].strip()
            else:
                answer = generated_text.strip()
        elif isinstance(result, dict) and "error" in result:
            answer = f"API Error: {result['error']}"
        else:
            answer = "No valid response from the API."
    except requests.exceptions.HTTPError as e:
        answer = f"HTTP Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        answer = f"Request Error: {str(e)}"
    except Exception as e:
        answer = f"Unexpected Error: {str(e)}"

    print("Sent to frontend:", answer)
    return {"response": answer}