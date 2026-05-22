from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as
      the prime minister of India since 2014.
      Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. 
      He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), 
      a right-wing Hindutva paramilitary volunteer organisation. 
      He is the longest-serving prime minister outside the Indian National Congress."""

    summary_template = """
    Given the information {info} about the person I want you to create: 
    1) A summary of his life
    2) 2 interesting facts about him
    """

    prompt = PromptTemplate(input_variables=["info"], template=summary_template)
    # llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = prompt | llm
    response = chain.invoke(input={"info": information})

    print(response.content)


if __name__ == "__main__":
    main()
