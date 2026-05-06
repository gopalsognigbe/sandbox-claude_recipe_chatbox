# RAG Recipe Chatbot (LangChain + LangGraph)

Chatbot RAG utilisant LangChain, LangGraph et HuggingFace LLM.

## Stack
- LangChain (orchestration)
- LangGraph (workflow/state management)
- HuggingFace (LLM backend)

## Architecture
- `recipes.json` : données
- `chatbot.py` : logique RAG + graph

## TODO
- [ ] Implémenter load_recipes()
- [ ] Implémenter search_recipes()
- [ ] Configurer HuggingFace LLM
- [ ] Créer la chaîne RAG avec LangChain
- [ ] Définir le graph LangGraph
- [ ] Tester avec des questions réelles