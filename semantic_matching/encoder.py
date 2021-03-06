import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss()
    
    def _init_embedding_weights(self):
        init_range = 0.5 / self.emb_dims
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, sentences1, sentences2, negatives=None, labels=None):
        in_batch_negative = labels is None
        sent1_emb = self.enocde_sentences(sentences1, True)
        if in_batch_negative:
            sent2_emb = self.enocde_sentences(sentences2, True)
            logits = torch.mm(sent1_emb, sent2_emb.permute(1, 0))
            if negatives is not None:
                neg_emb = self._encode_multiple_sentences(negatives)
                sent1_emb = sent1_emb.unsqueeze(dim=1)
                logits_ = torch.mul(sent1_emb, neg_emb).sum(dim=2)
                logits = torch.cat([logits, logits_], dim=1)
            batch_size = logits.shape[0]
            logits /= self.temperature
            labels = torch.arange(0, batch_size, device=logits.device)
        else:
            sent2_emb = self._encode_multiple_sentences(sentences2)
            sent1_emb = sent1_emb.unsqueeze(1)
            logits = torch.bmm(sent1_emb, sent2_emb.permute(0, 2, 1)).squeeze(1)
        return self.loss(logits, labels)

    def enocde_sentences(self, sentences, normalize=False):
        pass

    def _encode_multiple_sentences(self, sentences):
        batch_size, num_sents, seq_len = sentences.shape
        sentences = sentences.view(-1, seq_len)
        sent_emb = self.enocde_sentences(sentences)
        sent_emb = F.normalize(sent_emb, p=2, dim=1)
        sent_emb = sent_emb.view(batch_size, num_sents, -1)
        return sent_emb


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, attention_hidden_dims=100):
        super().__init__()
        self.in_dim = in_dim
        self.attention_hidden_dim = attention_hidden_dims
        self.projection = nn.Sequential(nn.Linear(in_dim, attention_hidden_dims), nn.Tanh())
        self.query = nn.Linear(attention_hidden_dims, 1, bias=False)

    def forward(self, x, masks=None):
        weights = self.query(self.projection(x)).squeeze(-1)
        if masks is not None:
            weights = weights.masked_fill(masks, -1e9)
        weights = torch.softmax(weights, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return output


class SiameseCbowEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, seq_len, temperature=0.05, embedding_weights=None, 
            pooling="mean", attention_hidden_dims=100):
        super().__init__(temperature)
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
            self._init_embedding_weights()
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AdditiveAttention(emb_dims, attention_hidden_dims)
        elif pooling == "full_connection":
            self.pool = nn.Linear(seq_len*emb_dims, emb_dims)

    def enocde_sentences(self, sentences, normalize=False):
        embedded = self.embedding(sentences)
        if self.pooling == "mean":
            encodings = embedded.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(embedded)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(embedded.reshape(batch_size, -1))
        if normalize:
            encodings = F.normalize(encodings)
        return encodings


class LstmEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, seq_len, temperature, embedding_weights=None,
            pooling="mean", attention_hidden_dims=100, dropout=0.2):
        super().__init__(temperature)
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
            self._init_embedding_weights()
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb_dims, emb_dims//2, num_layers=1,bidirectional=True, batch_first=True)
        self.drouput = nn.Dropout(dropout)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AdditiveAttention(emb_dims, attention_hidden_dims)
        elif pooling == "full_connection":
            self.pool = nn.Linear(seq_len*emb_dims, emb_dims)
    
    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.emb_dims // 2, device=device),
                torch.randn(2, batch_size, self.emb_dims // 2, device=device))
    
    def enocde_sentences(self, sentences, normalize=False):
        embeded = self.drouput(self.embedding(sentences))
        batch_size = sentences.shape[0]
        hidden = self.init_hidden(batch_size, sentences.device)
        lstm_out, hidden = self.lstm(embeded, hidden)
        if self.pooling == "mean":
            encodings = lstm_out.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(lstm_out)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(lstm_out.reshape(batch_size, -1))
        if normalize:
            encodings = F.normalize(encodings)
        return encodings


class MultiheadAttentionEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, num_heads, seq_len, temperature=0.05, embedding_weights=None, 
            pooling="mean", attention_hidden_dims=100, dropout=0.2):
        super().__init__(temperature)
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
            self._init_embedding_weights()
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.mha = nn.MultiheadAttention(emb_dims, num_heads, dropout=dropout)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AdditiveAttention(emb_dims, attention_hidden_dims)
        elif pooling == "full_connection":
            self.pool = nn.Linear(seq_len*emb_dims, emb_dims)
        self.dropout = nn.Dropout(dropout)

    def enocde_sentences(self, sentences, normalize=False):
        embedded = self.dropout(self.embedding(sentences))
        permuted = embedded.permute(1, 0, 2)
        attended, _ = self.mha(permuted, permuted, permuted)
        attended = attended.permute(1, 0, 2)
        if self.pooling == "mean":
            encodings = attended.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(attended)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(attended.reshape(batch_size, -1))
        if normalize:
            encodings = F.normalize(encodings)
        return encodings


class TransformerEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, num_heads, seq_len, num_layers=1, temperature=0.05, 
            embedding_weights=None, pooling="mean", attention_hidden_dims=100, dropout=0.2):
        super().__init__(temperature)
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
            self._init_embedding_weights()
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(emb_dims, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AdditiveAttention(emb_dims, attention_hidden_dims)
        elif pooling == "full_connection":
            self.pool = nn.Linear(seq_len*emb_dims, emb_dims)
        self.dropout = nn.Dropout(dropout)

    def enocde_sentences(self, sentences, normalize=False):
        embedded = self.dropout(self.embedding(sentences))
        encoded = self.transformer(embedded.permute(1, 0, 2))
        encoded = encoded.permute(1, 0, 2)
        if self.pooling == "mean":
            encodings = encoded.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(encoded)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(encoded.reshape(batch_size, -1))
        if normalize:
            encodings = F.normalize(encodings)
        return encodings


class BertEncoder(SentenceEncoder):
    def __init__(self, model_name_or_path, temperature=0.05, pooling="mean"):
        super().__init__(temperature)
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pooling = pooling

    def forward(self, sentences1, sentences2=None, negatives=None, labels=None):
        in_batch_negative = labels is None
        sent1_emb = self.enocde_sentences(sentences1, True)
        cse = sentences2 is None  # Contrastive learning
        if in_batch_negative:
            if cse:
                sentences2 = sentences1
            sent2_emb = self.enocde_sentences(sentences2, True)
            logits = torch.mm(sent1_emb, sent2_emb.permute(1, 0))
            if negatives:
                neg_emb = self.enocde_sentences(negatives, True)
                logits_ = torch.mul(sent1_emb, neg_emb).sum(dim=1, keepdim=True)
                logits = torch.cat([logits, logits_], dim=1)
            logits /= self.temperature
            batch_size = logits.shape[0]
            labels = torch.arange(0, batch_size, device=logits.device)
        else:
            sent2_emb = self.enocde_sentences(sentences2, True)
            logits = torch.mul(sent1_emb, sent2_emb).sum(dim=1)
        return self.loss(logits, labels)

    def enocde_sentences(self, sentences, normalize=False):
        output = self.bert_model(**sentences)
        if self.pooling == "mean":
            encoding = output.last_hidden_state.mean(dim=1)
        elif self.pooling == "cls":
            encoding = output.last_hidden_state[:, 0]
        if normalize:
            encoding = F.normalize(encoding, p=2, dim=1)
        return encoding


if __name__ == "__main__":
    # model = SiameseCbowEncoder(20, 30, 4, pooling="full_connection")
    # model = MultiheadAttentionEncoder(20, 30, 5, 4, pooling="full_connection")
    model = TransformerEncoder(20, 100, 5, 4, num_layers=2, pooling="mean")
    sents1 = torch.tensor([[1, 2, 4, 0], [2, 3, 4, 1]], dtype=torch.long)
    sents2 = torch.tensor([[1, 2, 4, 0], [2, 3, 4, 1]], dtype=torch.long)
    # negatives =  torch.tensor([[[1, 2, 4, 0], [2, 3, 4, 1]], [[1, 2, 4, 0], [2, 3, 4, 1]]], dtype=torch.long)
    print(model(sents1, sents2))

    # sents2 = torch.tensor([[[1, 2, 4, 0], [2, 3, 4, 1]], [[1, 2, 4, 1], [2, 3, 4, 1]]], dtype=torch.long)
    # labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    # print(model(sents1, sents2, labels))
    
    # model= BertEncoder("C:/code/models/chinese_base", pooling="cls")
    # sentences1 = model.tokenizer(["?????????", "?????????"], return_tensors="pt")
    # sentences2 = model.tokenizer(["?????????", "?????????"], return_tensors="pt")
    # negatives = model.tokenizer(["?????????", "?????????"], return_tensors="pt")
    # labels = torch.tensor([0, 1], dtype=torch.float)
    # print(model(sentences1, sentences2, negatives))
    # print(model(sentences1))
