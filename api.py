from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import load_recipes, create_llm, create_rag_chain, create_graph, ResponseCache

app = Flask(__name__)
CORS(app, origins="*")

# Init au démarrage
recipes = load_recipes("recipes.json")
llm     = create_llm()
chain   = create_rag_chain(llm)
cache   = ResponseCache(max_size=10)
graph   = create_graph(recipes, chain, cache)

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "query manquante"}), 400
    result = graph.invoke({"query": query, "context": "", "response": ""})
    return jsonify({"response": result["response"]})

if __name__ == "__main__":
    print("API démarrée sur http://localhost:8001")
    app.run(port=8001, debug=False)
