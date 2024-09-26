from fastapi import FastAPI
import openai
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("API_KEY")
vectara_api_key = os.getenv("VECTARA_API_KEY")
vectara_corpora_id = os.getenv("VECTARA_CORPUS_ID")
vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")

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

def optimize_with_gpt(vectara_response, question):
    conversation_history = [
        {"role": "system", "content": "You are an AI assistant. Your goal is to understand the user's question and provide a clear, accurate, and concise answer based on the information provided by Vectara."},
        {"role": "user", "content": f"User's question: {question}"},
        {"role": "system", "content": f"Vectara provided the following information relevant to the question: {vectara_response}"},
        {"role": "user", "content": "Please provide a clear and concise answer based on the information from Vectara, ensuring it fully addresses the user's question."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=0.1
    )

    if response and 'choices' in response:
        choices = response['choices']

        if choices and len(choices) > 0:
            optimized_response = choices[0]['message']['content']
            return optimized_response

    return "Sorry, something went wrong in optimizing the response."

# FastAPI route to handle the chatbot functionality
@app.post("/query")
async def handle_query(question: str):
    vectara_response = query_vectara(question)
    
    if vectara_response:
        optimized_response = optimize_with_gpt(vectara_response, question)
        return {"answer": optimized_response}
    else:
        return {"error": "Vectara did not find relevant information."}
