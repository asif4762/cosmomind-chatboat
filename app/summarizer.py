import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import ollama

load_dotenv()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        self.url = url
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irre in soup.body(["script", "style", "img", "input"]):
            irre.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


system_prompt = """
You are an assistant that analyzes the contents of a website 
and provides a short summary, ignoring text that might be navigation related.
Respond in markdown.
"""

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}\n"
    user_prompt += "The contents of this website are as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

MODEL = "llama3.2"

def summarize(url):
    website = Website(url)
    response = ollama.chat(model=MODEL, messages=messages_for(website))
    return response.message['content']

def display_summary(url):
    summary = summarize(url)
    return summary
