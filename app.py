import csv
import ollama
import time

with open("company_policies.csv", newline="", encoding="utf-8-sig") as file:
    policies = list(csv.DictReader(file))

print("Number of policies loaded:", len(policies))

question = input("Enter your company policy question: ")

policy_information = "\n".join(
    f"Policy: {p['title']}\nDetails: {p['policy_text']}"
    for p in policies
)
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

end_time = time.perf_counter()

print("\nQuestion:", question)
print("\nAI Answer:", response["message"]["content"])

response_time = end_time - start_time

print("\nResponse time:", round(response_time, 2), "seconds")
print("Input tokens:", response.prompt_eval_count)
print("Output tokens:", response.eval_count)
print("Total tokens:", (response.prompt_eval_count or 0) + (response.eval_count or 0))