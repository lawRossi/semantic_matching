from collections import defaultdict
from pycparser.ply.yacc import token
import torch
import jieba
import tqdm
import numpy as np
import lmdb
import pickle
import os
import shutil
import struct


class TextGroupDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, min_tf=3, max_len=30, vocab=None, tokenize=None, with_negative=False,
            cache_path=".sent_pair", rebuild_cache=False) -> None:
        super().__init__()
        self.data_file = data_file
        self.min_tf = min_tf
        self.max_len = max_len
        self.vocab = vocab
        self.tokenize = tokenize
        self.with_negative = with_negative

        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1  # account for word_dictionary
            if self.vocab is None:
                self.vocab = pickle.loads(txn.get(b"vocab"))
            self.vocab_size = len(self.vocab) + 2 # account for padding and oov

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(struct.pack(">I", index)))
        return item

    def __len__(self):
        return self.length

    def _load_data(self):
        self.data = []
        with open(self.data_file, encoding="utf-8") as fi:
            pbar = tqdm.tqdm(fi, "loading data")
            for line in pbar:
                splits = line.strip().split("\t")
                if self.with_negative:
                    sent1, sent2, *negatives = splits
                    self.data.append((sent1, sent2, negatives))
                elif len(splits) == 3:
                    label, sent1, sent2 = splits
                    self.data.append((sent1, sent2, int(label)))
                elif len(splits) == 2:
                    sent1, sent2 = splits
                    self.data.append((sent1, sent2))
                else:
                    raise RuntimeError("invalid file format")

    def _tokenize_sentences(self):
        tokenize = self.tokenize
        if tokenize is None:
            tokenize = jieba.lcut

        pbar = tqdm.trange(len(self.data), desc="processing tokenization")
        for i in pbar:
            item = self.data[i]
            sent1 = item[0]
            tokens1 = tokenize(sent1)
            sent2 = item[1]
            tokens2 = tokenize(sent2)
            if self.with_negative:
                tokens3 = [tokenize(sent) for sent in item[2]]
                new_item = (tokens1, tokens2, tokens3)
            elif len(item) == 2:
                new_item = (tokens1, tokens2)
            else:
                label = item[2]
                new_item = (tokens1, tokens2, label)
            self.data[i] = new_item

    def _build_vocabulary(self):
        counts = defaultdict(int)
        pbar = tqdm.tqdm(self.data, "building vocabulary")
        for item in pbar:
            for token in item[0]:
                counts[token] += 1
            for token in item[1]:
                counts[token] += 1
        words = [word for word, count in counts.items() if count >= self.min_tf]
        self.vocab = {word: i+1 for i, word in enumerate(words)}  # 0 kept for padding

    def _build_cache(self, cache_path):
        self._load_data()
        self._tokenize_sentences()
        if self.vocab is None:
            self._build_vocabulary()
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b"vocab", pickle.dumps(self.vocab))
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)

    def _yield_buffer(self):
        pbar = tqdm.trange(len(self.data), desc="build cache")
        buffer = []
        for i in pbar:
            item = self.data[i]
            token_idxes1 = self._token2idx(item[0])
            token_idxes2 = self._token2idx(item[1])
            if self.with_negative:
                token_idxes3 = self._token2idx(item[2])
                new_item = (token_idxes1, token_idxes2, token_idxes3)
            elif len(item) == 2:
                new_item = (token_idxes1, token_idxes2)
            else:
                label = item[2]
                new_item = (token_idxes1, token_idxes2, label)
            buffer.append((struct.pack(">I", i), pickle.dumps(new_item)))
            if len(buffer) % 1000 == 0:
                yield buffer
                buffer.clear()
        yield buffer

    def _token2idx(self, tokens):
        if isinstance(tokens[0], list):
            token_idxes = []
            for tokens_ in tokens:
                token_idxes_ = [self.vocab.get(token, len(self.vocab)+1) for token in tokens_]
                token_idxes_ = token_idxes_[:self.max_len] + [0] * (self.max_len - len(token_idxes_))
                token_idxes.append(token_idxes_)
        else:
            token_idxes = [self.vocab.get(token, len(self.vocab)+1) for token in tokens]
            token_idxes = token_idxes[:self.max_len] + [0] * (self.max_len - len(token_idxes))
        return np.array(token_idxes, dtype=np.int64)


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_len=30, cache_path=".bert", rebuild_cache=False) -> None:
        super().__init__()
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_len = max_len

        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(struct.pack(">I", index)))
        return item

    def __len__(self):
        return self.length
    
    def _build_cache(self, cache_path):
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)

    def _yield_buffer(self):
        buffer = []
        with open(self.data_file, encoding="utf-8") as fi:
            pbar = tqdm.tqdm(fi, "building cache")
            for i, line in enumerate(pbar):
                splits = line.strip().split("\t")
                if len(splits) == 3:
                    label, sent1, sent2 = splits
                elif len(splits) == 2:
                    sent1, sent2 = splits
                    label = None
                elif len(splits) == 1:
                    sent1 = splits[0]
                    sent2 = None
                    label = None
                else:
                    raise RuntimeError("invalid file format")
                sample = {} if label is None else {"label": int(label)}
                output = self._tokenize(sent1)
                output = {f"sent1_{k}": np.array(v, dtype=np.int64) for k, v in output.items()}
                sample.update(output)
                if sent2:
                    output = self._tokenize(sent2)
                    output = {f"sent2_{k}": np.array(v, dtype=np.int64) for k, v in output.items()}
                    sample.update(output)
                buffer.append((struct.pack(">I", i), pickle.dumps(sample)))
                if len(buffer) == 1000:
                    yield buffer
                    buffer.clear()
            yield buffer

    def _tokenize(self, sentence):
        return self.tokenizer(sentence, padding="max_length", max_length=self.max_len, truncation=True)
