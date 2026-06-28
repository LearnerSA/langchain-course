from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


load_dotenv()

@tool
def get_product_price(product_name: str) -> int:
    """
    Get the price of a product by its name in the catalog.
    """

    mock_prices = {
        "laptop": 999,
        "smartphone": 699,
        "headphones": 199,
        "monitor": 299
    }
    print(f"Getting price for product: {product_name}")
    return mock_prices.get(product_name.lower(), -1)

@tool
def get_product_discount(price: int, discount_tier: str) -> float:
    """
    Get the discount for a product by its price and discount tier.
    Available discount tiers: "gold", "silver", "bronze"
    """

    discount_percentages = {
        "gold": 0.20,
        "silver": 0.10,
        "bronze": 0.05
    }
    discount = discount_percentages.get(discount_tier.lower(), 0.0)


    print(f"Getting discount for product with price {price}, tier: {discount_tier}")
    return round(price * (1 - discount), 2)

@traceable(name = "Agent loop complete", tags=["example"])
def run_agent(question:str):
    tools = [get_product_price, get_product_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
                content=(
                      "You are a helpful assistant that can answer questions about product prices and discounts."
                      "You have access to the product price and discount tools. Use them to answer questions."
                      "You MUST use the tools to answer questions. Dont assume any product prices or discounts.\n"
                      " If you don't know the answer, say 'I don't know'.\n"
                      " Prices of the products are mentioned in the get_product_price tool. Discounts are mentioned in the get_product_discount tool.\n"
                      "Only call the get_product_discount tool after you have called the get_product_price tool and have a valid price."
                      "If user does not give discount tier, ask them and dont assume."
                      "use the functions to do calculations and dont do any calculations yourself.")),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration ~~~~~~~  {iteration + 1}:")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("AI Response:", ai_message.content)
            return ai_message.content
        
        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found in tool_dict.")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print(f" ERROR max iterations reached. Stopping the agent.")



if __name__ == "__main__":
    run_agent("What is the price of the laptop after applying a silver discount?")