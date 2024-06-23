from flask import Flask, request, render_template, make_response, session, flash
from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
import jwt
import os
import secrets
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import re
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
from dotenv import load_dotenv
from embed import init_embed, get_embed_answer
from qdrant import init_qdrant, get_qdrant_answer
from lama import init_lama, get_lama_answer
from rag import get_rag_answer
import asyncio

# Ensure the secret key is set
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

dkl_fly = 0

app_flask = Flask(__name__)
app_flask.secret_key = "abc"
app_fast = FastAPI()

# OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPEN AI KEY")
client = OpenAI(api_key=openai.api_key)

class ChatMessage(BaseModel):
    content: str
    role: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str

class ChatResponseList(BaseModel):
    answers: List[str]

class Token(BaseModel):
    access_token: str
    token_type: str

fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": "fakehashedpassword",
        "disabled": False,
    }
}

templates = Jinja2Templates(directory="templates")

def nl2br(value: str) -> str:
    ret = value.replace("\n", "<br>")
    return value

templates.env.filters['nl2br'] = nl2br

def verify_password(fake_hashed_password, password):
    return fake_hashed_password == "fakehashedpassword"

def authenticate_user(fake_db, username: str, password: str):
    user = fake_db.get(username)
    if not user:
        return False
    if not verify_password(user["hashed_password"], password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app_fast.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def extract_urls(messages: List[ChatMessage]) -> List[str]:
    url_pattern = re.compile(r'https?://\S+')
    urls = []
    for message in messages:
        urls.extend(url_pattern.findall(message.content))
    return urls

async def fetch_url_content(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()

async def generate_response(messages: List[ChatMessage], contexts: List[str]) -> str:
    user_message_content = "\n".join([msg.content for msg in messages if msg.role == "user"])
    context_str = "\n\n".join(contexts)

    system_prompt = "You are a helpful assistant. Use the provided contexts to answer the user's question."
    if dkl_fly == 1:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'openai.ChatCompletion {e}')
            return str(e)
    else:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                top_p=0.2,
                max_tokens=1000,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f'client.chat.completions {e}')
            return str(e)

def ask_chat_question(question):
    if dkl_fly == 1:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ask_chat_question {e}')
            return str(e)
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            model="gpt-3.5-turbo",
            max_tokens=50,
            temperature=0.7,
        )
        answer = chat_completion.choices[0].message.content
        return answer.strip()

async def process_chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    verify_token(token)
    responses = []
    messages = request.messages
    urls = extract_urls(messages)

    if len(urls) != 0:
        try:
            contexts = []
            for url in urls:
                content = await fetch_url_content(url)
                contexts.append(content)
            response = await generate_response(messages, contexts)
            return ChatResponse(answer=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

    for msg in request.messages:
        answer = ask_chat_question(msg.content)
        responses.append(answer)

    ans = "\n".join(responses)
    return ChatResponse(answer=ans)

async def open_ai_chat(message):
    token = create_access_token(data={"sub": "testuser"})
    chat_request = ChatRequest(messages=[ChatMessage(role="user", content=message)])
    response = await process_chat(chat_request, token)
    return response

@app_flask.route('/submit', methods=['POST'])
def submit():
    try:
        message = request.form['message']
        response = asyncio.run(open_ai_chat(message))
        return render_template('ragindex.html', answer=response.answer)
    except Exception as e:
        print(f'submit {e}')
        flash(f"Error: {str(e)}")
        return render_template('ragindex.html', answer="Error occurred. Please try again.")

@app_flask.route('/', methods=['GET'])
def home():
    return render_template('ragindex.html')

@app_fast.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    try:
        return templates.TemplateResponse("ragindex.html", {"request": request})
    except Exception as e:
        return HTMLResponse(content=f"Error rendering template: {e}", status_code=500)

@app_fast.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        response = await process_chat(request, token)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")

@app_fast.post("/batch_chat", response_model=ChatResponseList)
async def batch_chat(requests: List[ChatRequest], token: str = Depends(oauth2_scheme)):
    try:
        responses = [await process_chat(req, token) for req in requests]
        return ChatResponseList(answers=[res.answer for res in responses])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch chat requests: {e}")

@app_fast.post("/rag_chat", response_model=ChatResponse)
async def rag_chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        verify_token(token)
        question = request.messages[0].content

        doc_texts = ["Put your document texts here."]  # Replace with actual document texts
        doc_embeddings = init_embed(client, doc_texts)

        relevant_docs = get_embed_answer(client, doc_embeddings, question)
        answer = ask_chat_question(question)

        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RAG chat request: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the app')
    parser.add_argument('--use-flask', action='store_true', help='Run the Flask app')
    parser.add_argument('--use-fastapi', action='store_true', help='Run the FastAPI app')
    args = parser.parse_args()

    if args.use_flask:
        app_flask.run(host='0.0.0.0', port=5000)
    elif args.use_fastapi:
        import uvicorn
        uvicorn.run(app_fast, host="0.0.0.0", port=8000)
    else:
        print("Please specify --use-flask or --use-fastapi")
