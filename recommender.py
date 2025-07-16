import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], model: str = 'avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)

    if not corpus:
        return candidate  # 防止 corpus 为空时报错

    # 按时间从新到旧排序
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)

    # 时间衰减权重：最近的论文权重大
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    time_decay_weight = torch.tensor(time_decay_weight, dtype=torch.float32)

    # 提取语句嵌入
    corpus_abstracts = [paper['data']['abstractNote'] for paper in corpus]
    candidate_abstracts = [paper.summary for paper in candidate]

    corpus_feature = encoder.encode(corpus_abstracts, convert_to_tensor=True)   # shape: [n_corpus, dim]
    candidate_feature = encoder.encode(candidate_abstracts, convert_to_tensor=True)  # shape: [n_candidate, dim]

    # 计算相似度：[n_candidate, n_corpus]
    sim = util.cos_sim(candidate_feature, corpus_feature)

    # 使用时间衰减加权后打分
    scores = torch.matmul(sim, time_decay_weight) * 10  # shape: [n_candidate]

    # 将分数赋值到 candidate 对象中
    for s, c in zip(scores, candidate):
        c.score = s.item()

    # 根据分数降序排列
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate
