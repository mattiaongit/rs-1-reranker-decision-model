"""Cheap first pass: static embeddings (model2vec, ~30 MB, CPU only) keep the k links
closest to the target so the reranker only scores those."""
import os

import numpy as np
from model2vec import StaticModel


class Shortlist:
    def __init__(self, model=None):
        self.model = StaticModel.from_pretrained(model or os.environ.get("SHORTLIST", "minishlab/potion-base-8M"))

    def top(self, query, docs, k=40):
        """Indices of the k docs most similar to query, best first."""
        if len(docs) <= k:
            return list(range(len(docs)))
        q = self.model.encode([query])[0]
        d = self.model.encode(docs)
        sims = d @ q / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-9)
        return np.argsort(-sims)[:k].tolist()
