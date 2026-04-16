from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge base
with open("kb.txt", "r") as f:
    kb_text = f.read()

# Split into chunks
chunks = [c.strip() for c in kb_text.split("\n") if c.strip() != ""]

# Create embeddings
embeddings = model.encode(chunks)

# 🔥 Confidence threshold
CONFIDENCE_THRESHOLD = 0.5


def ask_question(question):
    question_emb = model.encode([question])[0]

    scores = []

    for chunk, chunk_emb in zip(chunks, embeddings):
        score = cosine_similarity([question_emb], [chunk_emb])[0][0]
        scores.append((score, chunk))

    # Sort by score
    scores.sort(reverse=True)

    print("\nTop Retrieved Chunks:")
    for s, c in scores[:2]:
        print(f"Score: {s:.3f} | {c}")

    best_score, best_chunk = scores[0]

    # 🔥 Validation layer
    if best_score < CONFIDENCE_THRESHOLD:
        return "I don’t know based on available data."

    return best_chunk


# Main loop
if __name__ == "__main__":
    while True:
        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        if question.strip() == "":
            print("\nAnswer: Please ask a valid question.")
            continue

        answer = ask_question(question)
        print("\nAnswer:", answer)
