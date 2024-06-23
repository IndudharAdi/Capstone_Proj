import re
from flask import Flask, request, jsonify
from playwright.sync_api import sync_playwright
import openai
import os

def extract_urls(chat_message):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(chat_message)
    return urls

def scrape_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        browser.close()
    return content

def generate_answer(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

app = Flask(__name__)
API_KEY = os.getenv('sk-proj-YXc9ZzdAotPDje2RPkq0T3BlbkFJylyg7kdFH7fVvn1spTda')
AUTH_TOKEN = os.getenv('f00be7bc2e26e0cf00558721a839290e')

@app.route('/rag', methods=['POST'])
def rag():
    auth_token = request.headers.get('Authorization')
    if auth_token != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    chat_message = data.get('message', '')
    urls = extract_urls(chat_message)

    if urls:
        content = ""
        for url in urls:
            content += scrape_content(url)
        answer = generate_answer(content, API_KEY)
    else:
        answer = generate_answer(chat_message, API_KEY)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
 
