import json
import os
from collections import OrderedDict
from typing import Any, Optional, TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM

load_dotenv()


# ---------------------------------------------------------------------------
# 0. Cache — mémorise les 10 dernières questions/réponses
# ---------------------------------------------------------------------------

class ResponseCache:
    """Cache LRU simple : conserve les 10 dernières paires question/réponse."""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.max_size = max_size

    def get(self, query: str) -> Optional[str]:
        key = query.strip().lower()
        if key in self._cache:
            self._cache.move_to_end(key)   # LRU : marque comme récemment utilisé
            return self._cache[key]
        return None

    def set(self, query: str, response: str) -> None:
        key = query.strip().lower()
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = response
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # retire le plus ancien

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_recipes(filepath: str) -> List[Dict]:
    """Charge les recettes depuis le fichier JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Retrieval — recherche par mots-clés
# ---------------------------------------------------------------------------

def search_recipes(recipes: List[Dict], query: str) -> List[Dict]:
    """Retourne les recettes les plus pertinentes pour la query."""
    query_words = query.lower().split()
    scored = []
    for recipe in recipes:
        score = 0
        for word in query_words:
            if word in recipe["name"].lower():
                score += 3
            for ingredient in recipe["ingredients"]:
                if word in ingredient.lower():
                    score += 1
            if word in recipe.get("difficulty", "").lower():
                score += 1
        scored.append((score, recipe))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for s, r in scored if s > 0]
    return top[:3] if top else recipes[:2]   # fallback : premières recettes


def format_context(recipes: List[Dict]) -> str:
    lines = []
    for r in recipes:
        lines.append(
            f"• {r['name']} | Ingrédients : {', '.join(r['ingredients'])} | "
            f"Temps : {r['time']} min | Difficulté : {r['difficulty']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. LLM + chaîne LangChain
# ---------------------------------------------------------------------------

class QwenLLM(LLM):
    """Wrapper LangChain pour Qwen2.5-72B-Instruct via HuggingFace Inference (novita)."""
    client: Any

    @property
    def _llm_type(self) -> str:
        return "qwen_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        result = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return result.choices[0].message.content


def create_llm():
    """Qwen2.5-72B-Instruct via HuggingFace Inference API (provider novita, gratuit)."""
    from huggingface_hub import InferenceClient

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print("[LLM] Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference (novita)")
    client = InferenceClient(provider="novita", api_key=token)
    return QwenLLM(client=client)


def create_rag_chain(llm):
    """Construit la chaîne LangChain (LCEL) : prompt | LLM | parser."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    template = (
        "Tu es un assistant culinaire. Réponds en français à partir "
        "uniquement des recettes ci-dessous.\n\n"
        "Recettes :\n{context}\n\n"
        "Question : {question}\n"
        "Réponse :"
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    if llm:
        return prompt | llm | StrOutputParser()
    return None


# ---------------------------------------------------------------------------
# 4. LangGraph — graph retrieve → generate
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    query: str
    context: str
    response: str


def create_graph(recipes: List[Dict], chain, cache: "ResponseCache"):
    """
    Construit et compile le graph LangGraph avec deux nœuds :
      retrieve  → cherche les recettes pertinentes
      generate  → génère la réponse via la chaîne RAG (avec cache)
    """
    from langgraph.graph import StateGraph, END

    def retrieve(state: ChatState) -> ChatState:
        # Court-circuit : si la réponse est déjà en cache, on saute retrieve
        if cache.get(state["query"]) is not None:
            return state
        found = search_recipes(recipes, state["query"])
        context = format_context(found)
        print(f"  [retrieve] {len(found)} recette(s) sélectionnée(s)")
        return {**state, "context": context}

    def generate(state: ChatState) -> ChatState:
        cached = cache.get(state["query"])
        if cached is not None:
            print(f"  [cache] Réponse trouvée en cache ({len(cache)}/{cache.max_size})")
            return {**state, "response": cached}
        response = chain.invoke({"context": state["context"], "question": state["query"]})
        response = response.strip()
        cache.set(state["query"], response)
        return {**state, "response": response}

    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. Test avec 3 questions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Recipe Chatbot  —  LangChain + LangGraph")
    print("=" * 60)

    recipes = load_recipes("recipes.json")
    print(f"[OK] {len(recipes)} recettes chargées\n")

    llm = create_llm()
    chain = create_rag_chain(llm)
    cache = ResponseCache(max_size=10)
    app = create_graph(recipes, chain, cache)
    print("[OK] Graph compilé : retrieve → generate (cache activé)\n")

    questions = [
        "Quels ingrédients faut-il pour les pâtes carbonara ?",
        "Quelle est la recette la plus rapide à préparer ?",
        "Y a-t-il une recette avec des oignons ?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"Question {i} : {question}")
        result = app.invoke({"query": question, "context": "", "response": ""})
        print(f"Réponse   : {result['response']}")
        print("-" * 60)
