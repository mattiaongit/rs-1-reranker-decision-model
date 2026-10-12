"""Local Qwen3-Reranker scorer: (query, document) pairs -> relevance in [0, 1].

Prompt format follows the Qwen3-Reranker model card: the model answers
"yes"/"no" and the score is P(yes) from the final-token logits.
"""
import copy
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(self, model=None, batch_size=32, max_length=512):
        model = model or os.environ.get("RERANKER", "Qwen/Qwen3-Reranker-0.6B")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        t = time.time()
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.batch_size = batch_size
        self.max_length = max_length
        print(f"[reranker] {model} on {self.device} in {time.time() - t:.1f}s")

    def _encode(self, texts):
        room = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self.tok(texts, add_special_tokens=False, truncation=True, max_length=room)
        ids = [self.prefix_ids + x + self.suffix_ids for x in enc["input_ids"]]
        return self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def score_slow(self, instruction, query, docs):
        """Reference implementation: every pair encoded in full."""
        texts = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        out = []
        for i in range(0, len(texts), self.batch_size):
            logits = self.model(**self._encode(texts[i : i + self.batch_size])).logits[:, -1, :]
            out.extend(self._p_yes(logits))
        return out

    def _p_yes(self, logits):
        pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
        return torch.softmax(pair, dim=1)[:, 1].tolist()

    @torch.no_grad()
    def score(self, instruction, query, docs, max_doc_tokens=64):
        """Same scores as score_slow, but the shared prefix (system prompt, instruction,
        query) runs through the model once and its KV cache is reused for every doc."""
        head = self.prefix_ids + self.tok.encode(
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>:", add_special_tokens=False)
        head_t = torch.tensor([head], device=self.device)
        cache = self.model.model(head_t, use_cache=True).past_key_values
        # Only "no"/"yes" at the last token matter: skip the 151k-word vocabulary projection elsewhere.
        yes_no = self.model.lm_head.weight[[self.no, self.yes]]
        tails = [self.tok.encode(" " + d, add_special_tokens=False)[:max_doc_tokens] + self.suffix_ids for d in docs]
        out = []
        # Every doc in a batch gets its own copy of the query's cache: long queries (page text in
        # them) get smaller batches, keeping the copies at most ~6k tokens' worth (~0.7 GB).
        batch = max(1, min(self.batch_size, 6000 // len(head)))
        for i in range(0, len(tails), batch):
            chunk = tails[i : i + batch]
            n, width = len(chunk), max(map(len, chunk))
            ids = torch.full((n, width), self.tok.pad_token_id, device=self.device)
            mask = torch.zeros((n, len(head) + width), dtype=torch.long, device=self.device)
            mask[:, : len(head)] = 1
            for j, t in enumerate(chunk):  # right-pad so positions continue from the head
                ids[j, : len(t)] = torch.tensor(t)
                mask[j, len(head) : len(head) + len(t)] = 1
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(n)
            hidden = self.model.model(ids, attention_mask=mask, past_key_values=c, use_cache=True).last_hidden_state
            last = torch.tensor([len(t) - 1 for t in chunk], device=self.device)
            pair = (hidden[torch.arange(n, device=self.device), last] @ yes_no.T).float()
            out.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
            del c, hidden
        del cache
        self.calls = getattr(self, "calls", 0) + 1
        if self.device == "mps" and self.calls % 8 == 0:
            torch.mps.empty_cache()  # hand cached GPU blocks back now and then; otherwise they pile up
        return out


_shared = {}


def shared(model=None, batch_size=16):
    """One loaded reranker per process, shared by every decision mode (it holds ~1.2 GB)."""
    model = model or os.environ.get("RERANKER", "Qwen/Qwen3-Reranker-0.6B")
    if model not in _shared:
        _shared[model] = Reranker(model, batch_size=batch_size)
    return _shared[model]
