import ollama

MODEL = "mistral:7b-instruct"

while True:
    question = input("\n❓ Ask something (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    response = ollama.generate(
        model=MODEL,
        prompt=question
    )

    print("\n🤖 Answer:")
    print(response["response"])