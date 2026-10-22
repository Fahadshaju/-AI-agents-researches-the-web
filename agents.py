from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()

#model
llm = ChatOllama(
    model="gemma4:12B",
        temperature=0,
        disable_streaming=True)

#1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )
    
#2nd agent
def build_scrape_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )
    
#write_chain
Writer_prompt = ChatPromptTemplate.from_messages([
        ("system","your a  epert researcher and you have to write a clear ,structred and insightful report on the given topic"),
        ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain= Writer_prompt | llm | StrOutputParser()

# critic_prompt
critics_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:"""
    ),
])

critic_chain = critics_prompt | llm | StrOutputParser()