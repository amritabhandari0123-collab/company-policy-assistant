import json
import math
import time
import ollama

# Load the saved vector index
with open("policy_vectors.json", "r", encoding="utf-8") as file:
    policies = json.load(file)

print("Number of indexed policies:", len(policies))

question = input("Enter your company policy question: ")

start_time = time.perf_counter()

# Convert the question into a vector
result = ollama.embed(
    model="nomic-embed-text",
    input=question
)
question_vector = result["embeddings"][0]

# Calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        return 0
    return dot_product / (length_a * length_b)

# Find the three most relevant policies
ranked_policies = sorted(
    policies,
    key=lambda p: cosine_similarity(question_vector, p["embedding"]),
    reverse=True
)
top_policies = ranked_policies[:3]

policy_information = "\n".join(
    f"Policy: {p['title']}\nDetails: {p['policy_text']}"
    for p in top_policies
)

# Ask Llama to answer using only the retrieved policies
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

end_time = time.perf_counter()

print("\nQuestion:", question)
print("\nRetrieved policies:")
for policy in top_policies:
    print("-", policy["title"])

print("\nAI Answer:", response["message"]["content"])
print("\nResponse time:", round(end_time - start_time, 2), "seconds")
print("Input tokens:", response.prompt_eval_count)
print("Output tokens:", response.eval_count)
print("Total tokens:", (response.prompt_eval_count or 0) + (response.eval_count or 0))