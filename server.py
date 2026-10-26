import csv
import time
import ollama
import json
import math
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

with open("company_policies.csv", newline="", encoding="utf-8-sig") as file:
    policies = list(csv.DictReader(file))

policy_information = "\n".join(
    f"Policy: {p['title']}\nDetails: {p['policy_text']}"
    for p in policies
)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/company_policies.csv")
def policy_csv():
    return send_from_directory(".", "company_policies.csv")

@app.route("/api/status")
def status():
    return jsonify({
        "status": "success",
        "message": "Company Policy Assistant backend is working"
    })

@app.route("/api/method2", methods=["POST"])
def method2():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    start_time = time.perf_counter()

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a company policy assistant. "
                    "Answer questions using only the company policies provided. "
                    "Identify the relevant policy by name. "
                    "If the policies do not contain the answer, say so."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Company policies:\n{policy_information}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    elapsed = time.perf_counter() - start_time
    input_tokens = response.prompt_eval_count or 0
    output_tokens = response.eval_count or 0

    return jsonify({
        "method": "LLM without vector index",
        "answer": response.message.content,
        "response_time_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    })
with open("policy_vectors.json", "r", encoding="utf-8") as file:
    indexed_policies = json.load(file)

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        return 0
    return dot_product / (length_a * length_b)

@app.route("/api/method3", methods=["POST"])
def method3():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    start_time = time.perf_counter()

    embedding_result = ollama.embed(
        model="nomic-embed-text",
        input=question
    )
    question_vector = embedding_result["embeddings"][0]

    ranked_policies = sorted(
        indexed_policies,
        key=lambda p: cosine_similarity(question_vector, p["embedding"]),
        reverse=True
    )
    top_policies = ranked_policies[:3]

    policy_information = "\n".join(
        f"Policy: {p['title']}\nDetails: {p['policy_text']}"
        for p in top_policies
    )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a company policy assistant. "
                    "Answer using only the provided policies. "
                    "Identify the relevant policy by name. "
                    "If the policies do not contain the answer, say so."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Company policies:\n{policy_information}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    elapsed = time.perf_counter() - start_time
    input_tokens = response.prompt_eval_count or 0
    output_tokens = response.eval_count or 0

    return jsonify({
        "method": "LLM with vector index",
        "answer": response.message.content,
        "relevant_policies": [p["title"] for p in top_policies],
        "response_time_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)