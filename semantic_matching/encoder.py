from os import sep
from numpy.lib.polynomial import poly
import torch
import torch.nn as nn
from collections import defaultdict
from itertools import chain
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from .dataset import SentencePairDataset
import argparse
import os.path


class SentenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, *input):
        return super().forward(*input)
    
    def forward(self, sentences1, sentences2, labels=None):
        in_batch_negative = labels is None
        sent1_emb = self.enocde_sentences(sentences1)
        if in_batch_negative:
            sent2_emb = self.enocde_sentences(sentences2)
            logits = torch.mm(sent1_emb, sent2_emb.permute(1, 0))
            batch_size = sentences1.shape[0]
            labels = torch.eye(batch_size, dtype=torch.float, device=sentences1.device)
        else:
            batch_size, num_sents, seq_len = sentences2.shape
            sentences2 = sentences2.view(-1, seq_len)
            sent2_emb = self.enocde_sentences(sentences2)
            sent2_emb = sent2_emb.view(batch_size, num_sents, -1)
            sent1_emb = sent1_emb.unsqueeze(1)
            print(sent1_emb.shape)
            print(sent2_emb.shape)
            logits = torch.bmm(sent1_emb, sent2_emb.permute(0, 2, 1)).squeeze(1)
        return self.loss(logits, labels)

    def enocde_sentences(self, sentences):
        pass


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
    def __init__(self, vocab_size, emb_dims, seq_len, embedding_weights=None, pooling="mean", attention_hidden_dims=100):
        super().__init__()
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AdditiveAttention(emb_dims, attention_hidden_dims)
        elif pooling == "full_connection":
            self.pool = nn.Linear(seq_len*emb_dims, emb_dims)

    def enocde_sentences(self, sentences):
        embedded = self.embedding(sentences)
        if self.pooling == "mean":
            encodings = embedded.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(embedded)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(embedded.reshape(batch_size, -1))
        return encodings


class MultiheadAttentionEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, num_heads, seq_len, embedding_weights=None, pooling="mean", 
            attention_hidden_dims=100, dropout=0.2):
        super().__init__()
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
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

    def enocde_sentences(self, sentences):
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
        return encodings


class TransformerEncoder(SentenceEncoder):
    def __init__(self, vocab_size, emb_dims, num_heads, seq_len, num_layers=1, embedding_weights=None, pooling="mean", 
            attention_hidden_dims=100, dropout=0.2):
        super().__init__()
        self.emb_dims = emb_dims
        if embedding_weights is None:
            self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
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

    def enocde_sentences(self, sentences):
        embedded = self.dropout(self.embedding(sentences))
        encoded = self.transformer(embedded)
        if self.pooling == "mean":
            encodings = encoded.mean(dim=1)
        elif self.pooling == "attention":
            encodings = self.pool(encoded)
        else:
            batch_size = sentences.shape[0]
            encodings = self.pool(encoded.reshape(batch_size, -1))
        return encodings


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="siamese_cbow", 
        choices=["siamese_cbow", "multihead_attention", "transformer"],  help="specify which encoder to use")
    parser.add_argument("--data_file", type=str, help="path of the data file")
    parser.add_argument("--save_dir", type=str, help="path of the directory to save model")
    parser.add_argument("--min_tf", type=int, default=3, help="minimum term frequence")
    parser.add_argument("--max_len", type=int, default=30, help="maximum number of tokens per sentence")
    parser.add_argument("--emb_dims", type=int, default=100, help="embedding dimensions")
    parser.add_argument("--num_heads", type=int, default=5, help="number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="number of transformer layers")
    parser.add_argument("--pooling", type=str, default="mean", choices=['mean', 'attention', 'full_connection'], 
        help="type of pooling, incuding 'mean', 'attention', 'full_connection'")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="number of epochs")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--device", type=str, default="cpu", help="specify which device to use")
    parser.add_argument("--use_tb", action="store_true", help="wether to use tensorboard")
    return parser


def train():
    parser = setup_argparser()
    args = parser.parse_args()
    dataset = SentencePairDataset(args.data_file, args.min_tf, args.max_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    vocab_size = len(dataset.vocab) + 2
    if args.encoder == "siamese_cbow":
        model = SiameseCbowEncoder(vocab_size, args.emb_dims, args.max_len, pooling=args.pooling)
    elif args.encoder == "multihead_attention":
        model = MultiheadAttentionEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, pooling=args.pooling)
    else:
        model = TransformerEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, args.num_layers, pooling=args.pooling)
    device = torch.device(args.device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.use_tb:
        tb_writer = SummaryWriter()

    global_step = 0
    log_interval = 50
    for _ in trange(args.epochs, desc="Epoch"):
        total_loss = 0
        pbar = tqdm(data_loader, desc="Interation")
        for step, batch in enumerate(pbar):
            batch = [item.to(device) for item in batch]
            loss = model(*batch)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                pbar.set_postfix(loss=total_loss/log_interval)
                total_loss = 0
            if args.use_tb:
                tb_writer.add_scalar("loss", loss.item(), global_step)
            global_step += 1

    weights = model.embedding.weight.cpu().detach().numpy()
    save_file = os.path.join(args.save_dir, "word_vecs.txt")
    with open(save_file, "w", encoding="utf-8") as fo:
        vocab = dataset.vocab
        fo.write(f"{len(vocab)+1} {args.emb_dims}\n")
        for word, idx in vocab.items():
            fo.write(f"{word} {' '.join('%.7f' % value for value in weights[idx])}\n")
        fo.write(f"#OOV# {' '.join('%.7f' % value for value in weights[-1])}\n")


if __name__ == "__main__":
    # model = SiameseCbowEncoder(20, 30, 4, pooling="full_connection")
    # model = MultiheadAttentionEncoder(20, 30, 5, 4, pooling="full_connection")
    # model = TransformerEncoder(20, 30, 5, 4, num_layers=2, pooling="full_connection")
    # sents1 = torch.tensor([[1, 2, 4, 0], [2, 3, 4, 1]], dtype=torch.long)
    # sents2 = torch.tensor([[1, 2, 4, 0], [2, 3, 4, 1]], dtype=torch.long)
    # print(model(sents1, sents2))

    # sents2 = torch.tensor([[[1, 2, 4, 0], [2, 3, 4, 1]], [[1, 2, 4, 1], [2, 3, 4, 1]]], dtype=torch.long)
    # labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    # print(model(sents1, sents2, labels))
    
    train()
