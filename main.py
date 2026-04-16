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

# 🔥 Thresholds
CONFIDENCE_THRESHOLD = 0.5
ANSWER_SIM_THRESHOLD = 0.6


# ✅ Retrieval function (MISSING in your code)
def retrieve(query, top_k=2):
    query_emb = model.encode([query])[0]
    
    scores = []
    for chunk, emb in zip(chunks, embeddings):
        score = cosine_similarity([query_emb], [emb])[0][0]
        scores.append((score, chunk))
    
    scores.sort(reverse=True)
    return scores[:top_k]


def ask_question(query):
    results = retrieve(query)
    
    print("\nTop Retrieved Chunks:")
    for score, chunk in results:
        print(f"Score: {score:.3f} | {chunk}")
    
    best_score, best_chunk = results[0]

    # ❌ Step 1: retrieval validation
    if best_score < CONFIDENCE_THRESHOLD:
        return "I don’t know based on available data."

    # ✅ Step 2: simulated answer (later replace with LLM)
    answer = best_chunk

    # 🔥 Step 3: answer-context similarity
    answer_emb = model.encode(answer)
    context_emb = model.encode(best_chunk)

    similarity = cosine_similarity([answer_emb], [context_emb])[0][0]

    print(f"\nAnswer-Context Similarity: {similarity:.3f}")

    # ❌ Step 4: validation
    if similarity < ANSWER_SIM_THRESHOLD:
        return "Generated answer is not grounded in context."

    return answer


# ✅ Clean main loop (ONLY ONCE)
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question: ")
        
        if q.lower() == "exit":
            break
        
        if q.strip() == "":
            print("\nAnswer: Please ask a valid question.")
            continue
        
        ans = ask_question(q)
        print("\nAnswer:", ans)
