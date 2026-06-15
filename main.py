from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()
@tool
def search(query: str) -> str:
   """Searches on the internet for the given query and returns the result.
    Args:
        query (str): The search query.
    Returns:
        str: The search result.
   """
   print(f"Searching for: {query}")
   return tavily.search(query=query)

def main():
    print("Starting the agent...")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    tools = [search]
    agent = create_agent(
        model=llm,
        tools=tools
    )
    
    response = agent.invoke({"messages": [HumanMessage(content="List 3 AI engineer job openings in Bangalore and list their details")]})
    print(response)
    


if __name__ == "__main__":
    main()
