import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
import sys

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()

    candidate_texts = [paper.summary for paper in candidate]
    corpus_texts = [p['data']['abstractNote'] for p in corpus]
    candidate_feature = encoder.encode(candidate_texts, convert_to_tensor=True)
    corpus_feature = encoder.encode(corpus_texts, convert_to_tensor=True)

    print("candidate_feature shape:", candidate_feature.shape)
    print("corpus_feature shape:", corpus_feature.shape) 
    sys.stdout.flush()

    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate
