from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI and Vectara API keys from environment variables
openai.api_key = os.getenv("API_KEY")
vectara_api_key = os.getenv("VECTARA_API_KEY")
vectara_corpora_id = os.getenv("VECTARA_CORPUS_ID")
vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")

# Initialize conversation history as a list
# Stores dictionaries with 'role' and 'content'
conversation_history = []
print("Initial conversation history:", conversation_history)

# Define the maximum number of messages to store
MAX_MESSAGES = 60  # 30 user messages + 30 assistant messages

class QueryRequest(BaseModel):
    question: str

def update_conversation_history(role: str, content: str):
    """
    Update the conversation history with a new message.
    Only 'user' and 'assistant' roles are stored.
    If the history exceeds MAX_MESSAGES, remove the oldest two messages.
    """
    if role not in ["user", "assistant"]:
        # Prevent adding system messages to conversation history
        return
    conversation_history.append({"role": role, "content": content})
    print(f"Added to conversation history: {{'role': '{role}', 'content': '{content}'}}")
    print(f"Current conversation history length: {len(conversation_history)}")

    # Maintain the maximum number of messages
    if len(conversation_history) > MAX_MESSAGES:
        # Remove the first two messages (oldest user and assistant messages)
        removed_user = conversation_history.pop(0)
        removed_assistant = conversation_history.pop(0)
        print(f"Removed from conversation history: {removed_user}")
        print(f"Removed from conversation history: {removed_assistant}")
        print(f"New conversation history length: {len(conversation_history)}")

def search_in_conversation_history(question: str):
    """
    Search for a cached response in the conversation history.
    Looks for the question in 'user' messages and returns the corresponding 'assistant' response.
    """
    print(f"Searching for cached response for question: '{question}'")
    for i in range(len(conversation_history)):
        entry = conversation_history[i]
        if entry['role'] == 'user' and entry['content'].strip().lower() == question.strip().lower():
            # Check if there's a corresponding assistant response
            if i + 1 < len(conversation_history):
                assistant_entry = conversation_history[i + 1]
                if assistant_entry['role'] == 'assistant':
                    print("Cached response found in conversation history.")
                    return assistant_entry['content']
    print("No cached response found in conversation history.")
    return None

def query_vectara(question: str):
    """
    Query the Vectara API with the given question and return the top result's text.
    """
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

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            result = response.json()
            if 'responseSet' in result and result['responseSet']:
                top_result = result['responseSet'][0]['response'][0]
                vectara_text = top_result.get("text", None)
                print(f"Vectara response: {vectara_text}")
                return vectara_text
            else:
                print("No results found in Vectara response.")
                return None
        else:
            print(f"Vectara query failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception during Vectara query: {e}")
        return None

def optimize_with_gpt(conversation_history: list, vectara_response: str, question: str):
    """
    Generate an optimized response using OpenAI's GPT model.
    Incorporates Vectara response if available.
    """
    # Add the user's question to the conversation history
    update_conversation_history("user", question)

    # Prepare messages to send to OpenAI, including Vectara data as a system message
    messages = conversation_history.copy()  # Make a copy to avoid mutation

    if vectara_response:
        system_message = {
            "role": "system",
            "content": f"Vectara provided the following information: {vectara_response}"
        }
    else:
        system_message = {
            "role": "system",
            "content": "No relevant information found in Vectara. I'll assist you based on previous conversation."
        }

    messages.append(system_message)

    print("Sending messages to OpenAI:")
    for msg in messages:
        print(msg)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )
        print("OpenAI response received.")
    except Exception as e:
        print(f"Exception during OpenAI API call: {e}")
        return "I'm sorry, but I'm experiencing some issues right now."

    if response and 'choices' in response:
        choices = response['choices']
        if choices and len(choices) > 0:
            optimized_response = choices[0]['message']['content'].strip()
            print(f"Optimized response from OpenAI: {optimized_response}")

            # Add the assistant's response to the conversation history
            update_conversation_history("assistant", optimized_response)

            return optimized_response

    print("No valid response from OpenAI.")
    return "I'm sorry, but I couldn't process your request at the moment."

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handle incoming queries from users.
    """
    question = request.question.strip()
    print(f"\nReceived question: {question}")

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check if there's a cached response in the conversation history
    cached_response = search_in_conversation_history(question)
    if cached_response:
        print("Returning cached response.")
        print(f"Conversation history (last {len(conversation_history)} messages):")
        for msg in conversation_history:
            print(msg)
        return {"answer": cached_response}

    # Query Vectara for additional information
    vectara_response = query_vectara(question)

    # Generate an optimized response using GPT
    optimized_response = optimize_with_gpt(conversation_history, vectara_response, question)

    # Print the updated conversation history after processing the query
    print("Updated conversation history:")
    for msg in conversation_history:
        print(msg)

    return {"answer": optimized_response}
