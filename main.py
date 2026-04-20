from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

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
QUESTION_SIM_THRESHOLD = 0.5


# -------------------------
# RETRIEVAL
# -------------------------
def retrieve(query, top_k=2):
    query_emb = model.encode(query)

    scores = []
    for chunk, emb in zip(chunks, embeddings):
        score = cosine_similarity([query_emb], [emb])[0][0]
        scores.append((score, chunk))

    scores.sort(reverse=True)
    return scores[:top_k]


# -------------------------
# SMART CHUNK SELECTION
# -------------------------
def select_best_chunk(query, results):
    query_words = set(query.lower().split())

    best_chunk = None
    best_score = -1

    for score, chunk in results:
        chunk_words = set(chunk.lower().split())

        overlap = len(query_words.intersection(chunk_words))

        final_score = score + (0.1 * overlap)

        if final_score > best_score:
            best_score = final_score
            best_chunk = chunk

    return best_chunk


# -------------------------
# GENERATOR AGENT (Simulated LLM)
# -------------------------
def simulate_llm_answer(query, context_chunks):
    mode = random.choice(["correct", "mix", "distort"])

    if mode == "correct":
        return context_chunks[0]

    elif mode == "mix":
        return " ".join(context_chunks[:2])

    elif mode == "distort":
        text = context_chunks[0]
        text = text.replace("20", "30").replace("9 AM", "10 AM")
        return text


# -------------------------
# VALIDATOR AGENT
# -------------------------
def validator_agent(query, answer, best_chunk):
    query_emb = model.encode(query)
    answer_emb = model.encode(answer)
    context_emb = model.encode(best_chunk)

    context_sim = cosine_similarity([answer_emb], [context_emb])[0][0]
    question_sim = cosine_similarity([answer_emb], [query_emb])[0][0]

    print(f"\n[Validator] Context Similarity: {context_sim:.3f}")
    print(f"[Validator] Question Similarity: {question_sim:.3f}")

    if context_sim < ANSWER_SIM_THRESHOLD:
        return False, "Not grounded in context"

    if question_sim < QUESTION_SIM_THRESHOLD:
        return False, "Does not answer question"

    return True, "Valid answer"


# -------------------------
# MAIN PIPELINE
# -------------------------
def ask_question(query, top_k):
    results = retrieve(query, top_k)

    print("\nTop Retrieved Chunks:")
    for score, chunk in results:
        print(f"Score: {score:.3f} | {chunk}")

    best_chunk = select_best_chunk(query, results)
    best_score = results[0][0]

    # Step 1: Retrieval validation
    if best_score < CONFIDENCE_THRESHOLD:
        return "I don’t know based on available data.", None

    context_chunks = [chunk for _, chunk in results]

    # Step 2: Generator Agent
    answer = simulate_llm_answer(query, context_chunks)

    print("\n[Generator] Answer:", answer)

    # Step 3: Validator Agent
    is_valid, reason = validator_agent(query, answer, best_chunk)

    if not is_valid:
        print(f"[Validator Rejected]: {reason}")
        return "I don’t know based on validation.", best_chunk

    return answer, best_chunk


# -------------------------
# RUN LOOP
# -------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question: ")

        if q.lower() == "exit":
            break

        if q.strip() == "":
            print("\nAnswer: Please ask a valid question.")
            continue

        try:
            top_k = int(input("Enter number of chunks (top_k): "))
        except:
            print("Invalid input. Using default top_k = 2")
            top_k = 2

        ans, source = ask_question(q, top_k)

        print("\nFinal Answer:", ans)
        print("Source:", source)
