from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
import aiofiles
import PyPDF2
import uuid
import requests
import json
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()

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

# Initialize conversation history
conversation_history = []

# Directory to store uploaded PDFs
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY", "./uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Define the maximum number of messages to store
MAX_MESSAGES = 60  # 30 user messages + 30 assistant messages


class QueryRequest(BaseModel):
    question: Optional[str] = None


def update_conversation_history(role: str, content: str):
    """
    Update the conversation history with a new message.
    Only 'user' and 'assistant' roles are stored.
    If the history exceeds MAX_MESSAGES, remove the oldest two messages.
    """
    if role not in ["user", "assistant"]:
        return
    conversation_history.append({"role": role, "content": content})

    # Maintain the maximum number of messages
    if len(conversation_history) > MAX_MESSAGES:
        # Remove the first two messages (oldest user and assistant messages)
        conversation_history.pop(0)
        conversation_history.pop(0)


def is_vectara_question(question: str) -> bool:
    """
    Check if the question is related to Vectara. You can customize this logic to suit your needs.
    In this example, we'll assume that questions with the word 'search', 'document', or 'file' 
    should be sent to Vectara.
    """
    vectara_keywords = ["search", "document", "file", "retrieve", "find"]
    return any(keyword in question.lower() for keyword in vectara_keywords)


def query_vectara(question: str) -> Optional[str]:
    """
    Query the Vectara API with the given question and return the top result's text.
    """
    url = f"https://api.vectara.io/v1/query"
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
                return vectara_text
            else:
                return "No results found in Vectara response."
        else:
            return f"Vectara query failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception during Vectara query: {e}"


def ingest_pdf_to_vectara(pdf_text: str, document_id: str):
    """
    Ingest the extracted text from the PDF into Vectara for indexing.
    """
    url = f"https://api.vectara.io/v1/corpora/{vectara_corpora_id}/documents"
    headers = {
        "x-api-key": vectara_api_key,
        "Content-Type": "application/json",
        "customer-id": vectara_customer_id
    }

    document = {
        "corpusId": vectara_corpora_id,
        "documentId": document_id,  # Unique identifier for the document
        "text": pdf_text,
        "metadata": {
            "filename": document_id
        }
    }

    try:
        ingest_response = requests.post(url, headers=headers, data=json.dumps(document))
        if ingest_response.status_code in [200, 201]:
            return "Successfully ingested PDF content into Vectara."
        else:
            return f"Failed to ingest PDF into Vectara: {ingest_response.status_code} {ingest_response.text}"
    except Exception as e:
        return f"Exception during Vectara ingestion: {e}"


@app.post("/interact")
async def interact(
    file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None)
):
    """
    Unified endpoint to handle PDF uploads and conversational queries.

    - If a PDF file is uploaded, it processes the file.
    - If a question is provided, it handles the query.
    - If both are provided, it prioritizes handling the file upload.
    """
    if file:
        # Handle PDF Upload
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Validate MIME type
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

        # Enforce file size limit (10 MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}.pdf"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

        # Save the uploaded PDF to the upload directory
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                await out_file.write(file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF: {e}")

        # Extract text from the PDF
        try:
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

        # Ingest the extracted PDF text into Vectara
        vectara_ingestion_response = ingest_pdf_to_vectara(text, unique_filename)
        return {"message": vectara_ingestion_response}

    elif question:
        # Determine if the question should be sent to Vectara or OpenAI
        if is_vectara_question(question):
            vectara_response = query_vectara(question)
            if vectara_response:
                update_conversation_history("assistant", vectara_response)
                return {"answer": vectara_response}
        else:
            # Handle conversational query via OpenAI
            optimized_response = optimize_with_gpt(conversation_history, question)
            return {"answer": optimized_response}

    else:
        # No valid action provided
        raise HTTPException(status_code=400, detail="Invalid request. Provide either a PDF file or a question.")


def optimize_with_gpt(conversation_history: list, question: str):
    """
    Generate a response using OpenAI's GPT model.
    """
    # Add the user's question to the conversation history
    update_conversation_history("user", question)

    # Prepare messages to send to OpenAI
    messages = conversation_history.copy()

    system_message = {
        "role": "system",
        "content": "Proceeding with the user's question."
    }
    messages.append(system_message)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )
    except Exception as e:
        return f"Error during OpenAI API call: {e}"

    if response and 'choices' in response:
        choices = response['choices']
        if choices and len(choices) > 0:
            optimized_response = choices[0]['message']['content'].strip()
            update_conversation_history("assistant", optimized_response)
            return optimized_response

    return "Sorry, I couldn't process your request at the moment."
