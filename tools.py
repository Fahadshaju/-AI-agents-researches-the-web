from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from rich import print
from dotenv import load_dotenv


load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information."""
    
    result=tavily.search(query= query,max_result=5)
    out=[]
    for r in result['results']:
        out.append(f"Title:{r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")
    return "\n______________\n".join(out)



@tool
def scrape_url(url: str) -> str:
    """Scrape the content of a given URL and give me the clean text with deeper reading."""
    try:
        response = requests.get(url, timeout = 8 , headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style','nav','header','footer','aside']):
            tag.decompose()
        return soup.get_text(separator= "", strip=True)[:3000]
    except Exception as e:
        return f"Error scraping the URL: {str(e)}"
    

    
