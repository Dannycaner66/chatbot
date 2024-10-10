from fastapi import FastAPI
import openai
from dotenv import load_dotenv
import os
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("API_KEY")
vectara_api_key = os.getenv("VECTARA_API_KEY")
vectara_corpora_id = os.getenv("VECTARA_CORPUS_ID")
vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

conversation_history = []

def update_conversation_history(role, content):
    conversation_history.append({"role": role, "content": content})
    
    if len(conversation_history) > 20:
        conversation_history.pop(0)  

def search_in_conversation_history(question):
    for entry in conversation_history:
        if question.lower() in entry['content'].lower() and entry['role'] == 'system':
            return entry['content']
    return None

def query_vectara(question):
    url = "https://api.vectara.io/v1/query"
    headers = {
        "x-api-key": vectara_api_key,
        "Content-Type": "application/json",
        "customer-id": vectara_customer_id
    }
    
    body = {
        'query': [
            {
                'query': question,
                'start': 0,
                'numResults': 1,
                'corpusKey': [
                    {'customerId': vectara_customer_id, 'corpusId': vectara_corpora_id}
                ],
                'context_config': {
                    'sentences_before': 2,
                    'sentences_after': 2,
                    'start_tag': "%START_SNIPPET%",
                    'end_tag': "%END_SNIPPET%",
                }
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(body))
    if response.status_code == 200:
        result = response.json()
        if 'responseSet' in result and result['responseSet']:
            top_result = result['responseSet'][0]['response'][0]
            return top_result.get("text", None)
    else:
        print(f"Vectara query failed with status code {response.status_code}: {response.text}")
    
    return None

def optimize_with_gpt(conversation_history, vectara_response, question):
    update_conversation_history("user", question)
    
    if vectara_response:
        update_conversation_history("system", f"Vectara provided the following information: {vectara_response}")
    else:
        update_conversation_history("system", "No relevant information found in Vectara. I'll assist you based on previous conversation.")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history, 
        temperature=0.5  
    )

    if response and 'choices' in response:
        choices = response['choices']

        if choices and len(choices) > 0:
            optimized_response = choices[0]['message']['content']
            
            update_conversation_history("system", optimized_response)
            
            return optimized_response

    return "Sorry, something went wrong in optimizing the response."

@app.post("/query")
async def handle_query(question: str):
    cached_response = search_in_conversation_history(question)
    if cached_response:
        return {"answer": cached_response}
    
    vectara_response = query_vectara(question)
    
    optimized_response = optimize_with_gpt(conversation_history, vectara_response, question)
    
    return {"answer": optimized_response}
