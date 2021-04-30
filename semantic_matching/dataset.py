from collections import defaultdict
import torch
import jieba
import tqdm
import numpy as np


class SentencePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, min_tf=3, max_len=30, tokenize=None) -> None:
        super().__init__()
        self.data_file = data_file
        self.min_tf = min_tf
        self.max_len = max_len
        self.tokenize = tokenize
        self._load_data()
        self._tokenize_sentences()
        self._build_vocabulary()
        self._convert_data()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        self.data = []
        with open(self.data_file, encoding="utf-8") as fi:
            pbar = tqdm.tqdm(fi, "loading data")
            for line in pbar:
                splits = line.strip().split("\t")
                if len(splits) == 3:
                    label, sent1, sent2 = splits
                    self.data.append((sent1, sent2, label))
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
            if len(item) == 2:
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

    def _convert_data(self):
        pbar = tqdm.trange(len(self.data), desc="converting data")
        for i in pbar:
            item = self.data[i]
            tokens1 = item[0]
            token_idxes1 = [self.vocab.get(token, len(self.vocab)+1) for token in tokens1]
            token_idxes1 = token_idxes1[:self.max_len] + [0] * (self.max_len - len(token_idxes1))
            tokens2 = item[1]
            token_idxes2 = [self.vocab.get(token, len(self.vocab)+1) for token in tokens2]
            token_idxes2 = token_idxes2[:self.max_len] + [0] * (self.max_len - len(token_idxes2))
            if len(item) == 2:
                new_item = (np.array(token_idxes1, dtype=np.int64), np.array(token_idxes2, dtype=np.int64))
            else:
                label = item[2]
                new_item = (np.array(token_idxes1), np.array(token_idxes2), label)
            self.data[i] = new_item
