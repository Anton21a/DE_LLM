from pathlib import Path

import random

import chromadb
from agents import (
    Agent,
    FunctionTool,
    function_tool,
)

import dotenv
dotenv.load_dotenv()

MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"


def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )


chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
tarot_cards_rag = chroma_client.get_collection(name="tarot_cards_rag")


TAROT_DECK = [
    # Major Arcana
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", 
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit", 
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance", 
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
    # Cups
    "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups", 
    "Six of Cups", "Seven of Cups", "Eight of Cups", "Nine of Cups", "Ten of Cups", 
    "Page of Cups", "Knight of Cups", "Queen of Cups", "King of Cups",
    # Pentacles
    "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles", 
    "Five of Pentacles", "Six of Pentacles", "Seven of Pentacles", "Eight of Pentacles", 
    "Nine of Pentacles", "Ten of Pentacles", "Page of Pentacles", "Knight of Pentacles", 
    "Queen of Pentacles", "King of Pentacles",
    # Swords
    "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords", 
    "Six of Swords", "Seven of Swords", "Eight of Swords", "Nine of Swords", "Ten of Swords", 
    "Page of Swords", "Knight of Swords", "Queen of Swords", "King of Swords",
    # Wands
    "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands", 
    "Six of Wands", "Seven of Wands", "Eight of Wands", "Nine of Wands", "Ten of Wands", 
    "Page of Wands", "Knight of Wands", "Queen of Wands", "King of Wands"
]

@function_tool
def draw_tarot_cards_tool(topic: str = "general", n_cards: int = 3) -> str:
    """
    Draw random tarot cards for a user reading.
    
    Args:
        topic: The topic of the reading, e.g. love, career, general
        n_cards: Number of cards to draw
        
    Returns:
        A text description of drawn cards with their orientation (upright/reversed).
    """
    chosen = random.sample(TAROT_DECK, n_cards)
    
    results = []
    for i, card in enumerate(chosen, start=1):
        orientation = random.choice(["upright", "reversed"])
        results.append(f"{i}. {card} ({orientation})")
    
    return f"Tarot reading topic: {topic}\nDrawn cards:\n" + "\n".join(results)


@function_tool
def tarot_lookup_tool(card_name: str, max_results: int = 1) -> str:
    """
    Tool function for a RAG database to look up meanings and details for a specific Tarot card.

    Args:
        card_name: The name of the tarot card to look up (e.g. "The Fool").
        max_results: The maximum number of results to return.

    Returns:
        A string containing the tarot card information and meanings.
    """
    results = tarot_cards_rag.query(query_texts=[card_name], n_results=max_results)

    if not results["documents"][0]:
        return f"No information found for card: {card_name}"


    formatted_results = []
    for doc in results["documents"][0]:
        formatted_results.append(doc)

    return f"Meaning for {card_name}:\n" + "\n\n---\n\n".join(formatted_results)


tarot_agent = Agent(
    name="Tarot predictions",
    instructions="""
    You are a machine answering user's questions with the use of Tarot cards.
    You randomly take 3 cards out of 78, and then express their meanings together.
    The resulting prediction should make sense and be relevant to the question of the user.
    If you need to randomly choose cards, use the tool: draw_tarot_cards_tool
    If you need to look up cards meanings, use the tool: tarot_lookup_tool
    """,
    model=MODEL,
    tools=[
        bedrock_tool(draw_tarot_cards_tool.__dict__), 
        bedrock_tool(tarot_lookup_tool.__dict__)
    ],
)