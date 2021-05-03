from .index import AnnoyIndex
from itertools import chain
import random


def sample_hard_negative(encoder, sentence_groups, num_negative, index_dir="tmp"):
    index = AnnoyIndex(encoder, index_dir)
    sentences = list(chain.from_iterable(sentence_groups))
    documents = [
        {"id": i, "index_text": sentence} for i, sentence in enumerate(sentences)
    ]
    encodings = encoder.encode_sentences(sentences)
    index._build_index(documents, encodings)
    sentence_idx_groups = []
    start_idx = 0
    for group in sentence_groups:
        sentence_idx_groups.append(list(range(start_idx, start_idx+len(group))))
        start_idx += len(group)
    negative_samples = []
    for group in sentence_idx_groups:
        query_idx = random.choice(group)
        query_encoding = encodings[query_idx]
        topk = len(group) + 3 * num_negative
        retrieved = index.retrieve_by_encoding(query_encoding, topk)
        retrieved = [idx for idx in retrieved if idx not in group]
        negative_idxes = random.sample(retrieved, num_negative)
        negative_sentences = [sentences[idx] for idx in negative_idxes]
        negative_samples.append(negative_sentences)
    return negative_samples
