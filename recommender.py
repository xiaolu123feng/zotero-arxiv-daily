import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def prepare_texts(candidate, corpus):
    valid_candidate = []
    candidate_texts = []
    for paper in candidate:
        summary = getattr(paper, 'summary', None)
        if summary and isinstance(summary, str) and summary.strip():
            candidate_texts.append(summary.strip())
            valid_candidate.append(paper)
        else:
            print(f"Skipping invalid candidate summary")

    valid_corpus = []
    corpus_texts = []
    for p in corpus:
        abstract = p['data'].get('abstractNote')
        if abstract and isinstance(abstract, str) and abstract.strip():
            corpus_texts.append(abstract.strip())
            valid_corpus.append(p)
        else:
            print(f"Skipping invalid corpus abstract")

    return valid_candidate, candidate_texts, valid_corpus, corpus_texts



def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()

    candidate_texts = [paper.summary for paper in candidate]
    corpus_texts = [p['data']['abstractNote'] for p in corpus]

    candidate, candidate_texts, corpus, corpus_texts = prepare_texts(candidate, corpus)

    candidate_feature = encoder.encode(candidate_texts, convert_to_tensor=True)
    corpus_feature = encoder.encode(corpus_texts, convert_to_tensor=True)


    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate
