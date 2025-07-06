import requests
import os
from dotenv import load_dotenv


load_dotenv()
hf_api_token = os.getenv("HF_API_TOKEN")

if not hf_api_token:
    print("ERROR: HF_API_TOKEN not found in .env file. Please check your .env file.")
    exit(1)


model = "distilgpt2"  
api_url = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {hf_api_token}"}

def get_response(text):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 100,  
            "min_length": 10,   
            "temperature": 0.5,  
            "top_p": 0.9,       
            "do_sample": True   
        }
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Raw API Response:", result)
        
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            generated_text = result[0]["generated_text"]

            if generated_text.startswith(text):
                answer = generated_text[len(text):].strip()
            else:
                answer = generated_text.strip()
            
            answer_words = answer.split()
            seen = set()
            answer = " ".join(word for word in answer_words if not (word in seen or seen.add(word)))
            
            answer = answer.split('.')[0] + '.' if '.' in answer else answer
            return answer
        elif isinstance(result, dict) and "error" in result:
            return f"API Error: {result['error']}"
        else:
            return "No valid response from the API."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"Request Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

# Main loop
print("Welcome to Formation AI Terminal Chatbot!")
print(f"Using model: {model}")
print("Type your question below (or 'quit' to exit):")
while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    if not user_input:
        print("Please enter some text.")
        continue
    
    print("Processing...")
    answer = get_response(user_input)
    print("Response:", answer)