# RAG Recipe Chatbot — Context for Claude

## Architecture
- `chatbot.py` : logique RAG principale (LangChain + LangGraph)
- `api.py` : endpoint API pour le frontend
- `data/recipes.json` : source de données des recettes
- `index.html` : interface frontend

## Key rules
- Utiliser toujours HuggingFace pour le LLM (jamais OpenAI)
- Ne jamais modifier `recipes.json` directement
- Toujours utiliser `load_recipes()` pour accéder aux recettes
- Les tests doivent passer avant tout commit

## Stack technique
- LangChain + LangGraph pour l'orchestration
- HuggingFace Hub pour le LLM
- Python 3.10+

## Commandes utiles
- `python chatbot.py` : lancer le chatbot en mode CLI
- `python api.py` : démarrer l'API
- `python -m pytest tests/` : lancer les tests
## MCP Servers (extensions externes)
- Configuration dans `.claude/mcp/config.json`
- Serveurs disponibles : à définir