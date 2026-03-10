import sys
sys.path.insert(0, '.')

from rag.retriever import HybridRetriever

r = HybridRetriever()
chunks = r.retrieve('Wild Alaskan Salmon Omega-3', 'Nordic Naturals')
print('Chunks found:', len(chunks))
for c in chunks:
    print('Score:', c['score'], '| Text:', c['text'][:100])