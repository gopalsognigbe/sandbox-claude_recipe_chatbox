from langchain.llms import HuggingFaceLLM
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
import json

def load_recipes(filepath):
    """Charge les recettes depuis le JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def search_recipes(recipes, query):
    """Cherche les recettes pertinentes basées sur la query"""
    # TODO: implémenter une recherche simple par keywords
    pass

def create_rag_chain():
    """Crée la chaîne LangChain pour le RAG"""
    # TODO: utiliser HuggingFace LLM + prompt template
    pass

def create_graph():
    """Crée le graph LangGraph pour le workflow du chatbot"""
    # TODO: définir les nœuds et transitions
    pass

if __name__ == "__main__":
    # TODO: initialiser et tester le chatbot
    pass