from rag.retriever import HybridRetriever

r = HybridRetriever()
chunks = r.retrieve("Omega-3 Fish Oil")
print(f"Chunks found: {len(chunks)}")
for c in chunks:
    print(f"Score: {c['score']} | Text: {c['text'][:80]}")