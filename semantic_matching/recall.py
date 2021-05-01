from elasticsearch import Elasticsearch
import annoy
import os
import faiss
import json


class DocumentIndex:
    def build_index(self, documents):
        pass
    
    def load_index(self):
        pass

    def retrieve(self, query, max_num):
        pass
    
    def add_documents(self, documents):
        pass

    def get_document_by_ids(self, document_ids):
        pass


class ESIndex(DocumentIndex):
    def __init__(self, es, es_index="document"):
        self. es = es
        self.es_index = es_index

    def build_index(self, documents):
        self._check_index()
        for document in documents:
            self.es.index(self.es_index, document)

    def add_documents(self, documents):
        for document in documents:
            self.es.index(self.es_index, document)

    def _check_index(self):
        es = self.es
        if not es.indices.exists(self.es_index):
            es.indices.create(self.es_index)
            es.indices.close(self.es_index)
            setting = {
                "index.analysis.analyzer.default.type": "ik_max_word",
                "index.analysis.search_analyzer.default.type": "ik_samrt"
            }
            es.indices.put_settings(setting, index=self.es_index)
            es.indices.open(self.es_index)

    def retrieve(self, query, max_num):
        res = self.es.search({"query": {"match": {"index_text": query}}}, index=self.es_index, size=max_num)
        retrieved_documents = [hit["_source"] for hit in res["hits"]["hits"]]
        return retrieved_documents
    
    def get_document_by_ids(self, document_ids):
        pass


class AnnoyIndex(DocumentIndex):
    def __init__(self, encoder, index_dir="index", dimension=100, num_trees=50, metric="angular"):
        self.encoder = encoder
        self.annoy_index = None
        self.index_dir = index_dir
        self.dimension = dimension
        self.num_trees = num_trees
        self.metric = metric
        self.document_ids = []

    def build_index(self, documents):
        texts = [document["index_text"] for document in documents]
        encodings = self.encoder.encode_sentences(texts)
        self.annoy_index = annoy.AnnoyIndex(self.dimension, metric=self.metric)
        for i, (document, encoding) in enumerate(zip(documents, encodings)):
            self.annoy_index.add_item(i, encoding)
            self.document_ids.append(document["id"])
        self.annoy_index.build(self.num_trees)
        self._save_annoy_index()

    def load_index(self):
        with open(os.path.join(self.index_dir, "parameters.txt")) as fi:
            dimension = int(fi.readline())
        self.annoy_index = annoy.AnnoyIndex(dimension, metric=self.metric)
        index_file = os.path.join(self.index_dir, "index.ann")
        self.annoy_index.load(index_file)
        with open(os.path.join(self.index_dir, "document_ids.json")) as fi:
            self.document_ids = json.load(fi)

    def add_documents(self, documents):
        raise NotImplementedError

    def retrieve(self, query, max_num):
        query_vec = self.encoder.encode_sentences([query])[0]
        retrieved_ids = [self.document_ids[idx] for idx in self.annoy_index.get_nns_by_vector(query_vec, max_num)]
        return retrieved_ids

    def get_document_by_ids(self, document_ids):
        raise NotImplementedError

    def _save_annoy_index(self):
        save_dir = self.index_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        index_file = os.path.join(save_dir, "index.ann")
        self.annoy_index.save(index_file)
        with open(os.path.join(save_dir, "parameters.txt"), "w") as fo:
            fo.write(str(self.dimension)+"\n")
        with open(os.path.join(self.index_dir, "document_ids.json"), "w") as fo:
            json.dump(self.document_ids, fo)


class FaissIndex(DocumentIndex):
    def __init__(self, encoder, index_dir, dimension, n_clusters=1, n_pq=10, n_bytes=4, metric="cosine"):
        self.encoder = encoder
        self.index_dir = index_dir
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.n_pq = n_pq
        self.n_bytes = n_bytes
        self.metric = metric

    def _init_index(self):
        quantizer = faiss.IndexFlatIP(self.dimension) 
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.n_clusters, self.n_bytes, self.n_pq)

    def build_index(self, documents):
        self._init_index()
        texts = [document["index_text"] for document in documents]
        encodings = self.encoder.encode_sentences(texts)
        if self.metric == "cosine":
            faiss.normalize_L2(encodings)
        self.index.train(encodings)
        self.index.add(encodings)
        self.document_ids = [document["id"] for document in documents]
        self._save_index()

    def retrieve(self, query, max_num):
        query_vec = self.encoder.encode_sentences([query])
        if self.metric == "cosine":
            faiss.normalize_L2(query_vec)
        _, nns = self.index.search(query_vec, max_num)
        return [[self.document_ids[idx] for idx in item if idx != -1] for item in nns]

    def add_documents(self, documents):
        texts = [document["index_text"] for document in documents]
        encodings = self.encoder.encode(texts)
        self.index.add(encodings)
        for document in documents:
            self.document_ids.append(document["id"])

    def _save_index(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        index_file = os.path.join(self.index_dir, "index.faiss")
        faiss.write_index(self.index, index_file)
        with open(os.path.join(self.index_dir, "document_ids.json"), "w") as fo:
            json.dump(self.document_ids, fo)

    def load_index(self):
        index_file = os.path.join(self.index_dir, "index.faiss")
        self.index = faiss.read_index(index_file)
        with open(os.path.join(self.index_dir, "document_ids.json")) as fi:
            self.document_ids = json.load(fi)

    def get_document_by_ids(self, document_ids):
        raise NotImplementedError
