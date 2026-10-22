
from agents import (
    build_search_agent,
    build_scrape_agent,
    writer_chain,
    critics_prompt,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from rich import print

llm = ChatOllama(
    model="gemma4:12B",
    temperature=0,
    disable_streaming=True,
)

state = {}

def run_research_pipeline(topic: str) -> dict:
    """Run the research pipeline for a given topic."""

    # Step 1: Search the web
    search_agent = build_search_agent()

    search_results = search_agent.invoke({
        "messages": [
            ("user",
             f"Search the web for recent and reliable information "
             f"on the topic: {topic}.")
        ]
    })

    state["search_results"] = search_results["messages"][-1].content

    print(
        f"[bold green]Search Results:[/bold green]\n"
        f"{state['search_results'][:700]}...\n"
    )

    # Step 2: Scrape relevant content
    scrape_agent = build_scrape_agent()

    reader_results = scrape_agent.invoke({
        "messages": [
            ("user",
             f"Based on the following search about '{topic}', "
             f"pick the most relevant URL and scrape it for deeper content.\n\n"
             f"Search Results:\n{state['search_results'][:700]}")
        ]
    })

    state["scraped_content"] = reader_results["messages"][-1].content

    print(
        f"[bold green]Scraped Content:[/bold green]\n"
        f"{state['scraped_content'][:700]}...\n"
    )

    # Step 3: Write the research report
    research_combined = (
        f"Search results:\n{state['search_results']}\n\n"
        f"Detailed scraped content:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    print("\nFINAL REPORT\n", state["report"])

    # Step 4: Critique the report
    critic_chain = critics_prompt | llm | StrOutputParser()

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCRITIC REPORT\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter the research topic: ").strip()

    if topic:
        run_research_pipeline(topic)
    else:
        print("Please enter a research topic.")
