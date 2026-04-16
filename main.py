from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge base
with open("kb.txt", "r") as f:
    kb_text = f.read()

# Split into chunks
chunks = [c.strip() for c in kb_text.split("\n") if c.strip() != ""]

# Create embeddings
embeddings = model.encode(chunks)

# Thresholds
CONFIDENCE_THRESHOLD = 0.5
ANSWER_SIM_THRESHOLD = 0.6


def retrieve(query, top_k=2):
    query_emb = model.encode(query)

    scores = []
    for chunk, emb in zip(chunks, embeddings):
        score = cosine_similarity([query_emb], [emb])[0][0]
        scores.append((score, chunk))

    scores.sort(reverse=True)
    return scores[:top_k]


def generate_answer(query, context):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def ask_question(query):
    results = retrieve(query)

    print("\nTop Retrieved Chunks:")
    for score, chunk in results:
        print(f"Score: {score:.3f} | {chunk}")

    best_score, best_chunk = results[0]

    # Step 1: retrieval validation
    if best_score < CONFIDENCE_THRESHOLD:
        return "I don’t know based on available data.", None

    # Step 2: LLM generation
    answer = best_chunk

    # Step 3: answer-context similarity
    answer_emb = model.encode(answer)
    context_emb = model.encode(best_chunk)

    similarity = cosine_similarity([answer_emb], [context_emb])[0][0]

    print(f"\nAnswer-Context Similarity: {similarity:.3f}")

    # Step 4: validation
    if similarity < ANSWER_SIM_THRESHOLD:
        return "Generated answer is not grounded in context."

    return answer, best_chunk


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question: ")

        if q.lower() == "exit":
            break

        if q.strip() == "":
            print("\nAnswer: Please ask a valid question.")
            continue

        ans, source = ask_question(q)
        print("\nAnswer:", ans)
        print("Source:", source)
