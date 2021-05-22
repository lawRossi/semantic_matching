from .index import AnnoyIndex
from itertools import chain
import random
from collections import defaultdict
from sklearn.cluster import KMeans


def sample_hard_negative_with_index(encoder, sentence_groups, num_negative, index_dir="tmp", topk=30):
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
        retrieved = index.retrieve_by_encoding(query_encoding, topk)
        retrieved = [idx for idx in retrieved if idx not in group]
        negative_idxes = random.sample(retrieved, num_negative)
        negative_sentences = [sentences[idx] for idx in negative_idxes]
        negative_samples.append(negative_sentences)
    return negative_samples


def sample_hard_negatives_by_clustering(encoder, sentence_groups, num_negative, n_clusters):
    sentences = list(chain.from_iterable(sentence_groups))
    encodings = encoder.encode_sentences(sentences)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_indexes = kmeans.fit_prediction(encodings)
    text_cluster_indexes = {sentences[i]: idx for i, idx in enumerate(cluster_indexes)}
    cluster_group = defaultdict(list)
    for i, idx in enumerate(cluster_indexes):
        cluster_group[idx].append(i)
    negative_samples = []
    for group in sentence_groups:
        cluster_idx = text_cluster_indexes[random.choice(group)]
        candidates = random.sample(cluster_group[cluster_idx], 3 * num_negative)
        candidates = [sentences[idx] for idx in candidates if sentences[idx] not in group]
        candidates = candidates[:num_negative]
        if len(candidates) < num_negative:
            candidates = candidates + group[len(candidates)-num_negative:]
        negative_samples.append(candidates)
    return negative_samples
