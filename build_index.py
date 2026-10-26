import csv
import json
import ollama

# Step 1: Load company policies
with open("company_policies.csv", newline="", encoding="utf-8-sig") as file:
    policies = list(csv.DictReader(file))

print("Number of policies loaded:", len(policies))

# Step 2: Convert each policy into a vector
vector_index = []

for number, policy in enumerate(policies, start=1):
    text = f"{policy['title']}. {policy['policy_text']}"

    result = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    vector_index.append({
        "title": policy["title"],
        "policy_text": policy["policy_text"],
        "embedding": result["embeddings"][0]
    })

    print(f"Indexed policy {number}/{len(policies)}")

# Step 3: Save the vector index
with open("policy_vectors.json", "w", encoding="utf-8") as file:
    json.dump(vector_index, file)

print("Vector index saved successfully!")