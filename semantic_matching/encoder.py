import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from .dataset import BertDataset, SentencePairDataset
import argparse
import os.path
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import math
import json


class SentenceEncoder(nn.Module):
    def __init__(self, similarity_func="dot"):
        super().__init__()
        self.similarity_func = similarity_func
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, sentences1, sentences2, labels=None):
        in_batch_negative = labels is None
        sent1_emb = self.enocde_sentences(sentences1)
        if self.similarity_func == "cosine":
            sent1_emb = normalize(sent1_emb, p=2, dim=1)
        if in_batch_negative:
            sent2_emb = self.enocde_sentences(sentences2)
            if self.similarity_func == "cosine":
                sent2_emb = normalize(sent2_emb, p=2, dim=1)
            logits = torch.mm(sent1_emb, sent2_emb.permute(1, 0))
            batch_size = sentences1.shape[0]
            labels = torch.eye(batch_size, dtype=torch.float, device=sentences1.device)
        else:
            batch_size, num_sents, seq_len = sentences2.shape
            sentences2 = sentences2.view(-1, seq_len)
            sent2_emb = self.enocde_sentences(sentences2)
            if self.similarity_func == "cosine":
                sent2_emb = normalize(sent2_emb, p=2, dim=1)
            sent2_emb = sent2_emb.view(batch_size, num_sents, -1)
            sent1_emb = sent1_emb.unsqueeze(1)
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
    def __init__(self, vocab_size, emb_dims, seq_len, embedding_weights=None, pooling="mean", similarity_func="dot", attention_hidden_dims=100):
        super().__init__(similarity_func)
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
            similarity_func="dot", attention_hidden_dims=100, dropout=0.2):
        super().__init__(similarity_func)
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
            similarity_func="dot", attention_hidden_dims=100, dropout=0.2):
        super().__init__(similarity_func)
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


class BertEncoder(SentenceEncoder):
    def __init__(self, model_name_or_path, similarity_func="dot"):
        super().__init__(similarity_func)
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def forward(self, sentences1, sentences2=None, labels=None):
        in_batch_negative = labels is None
        sent1_emb = self.enocde_sentences(sentences1)
        cse = sentences2 is None  # Contrastive learning
        if self.similarity_func == "cosine" or cse:
            sent1_emb = normalize(sent1_emb, p=2, dim=1)
        if in_batch_negative:
            if cse:
                sentences2 = sentences1
            sent2_emb = self.enocde_sentences(sentences2)
            if self.similarity_func == "cosine" or cse:
                sent2_emb = normalize(sent2_emb, p=2, dim=1)
            logits = torch.mm(sent1_emb, sent2_emb.permute(1, 0))
            if cse:
                logits *= 20
            batch_size = logits.shape[0]
            labels = torch.eye(batch_size, dtype=torch.float, device=logits.device)
        else:
            sent2_emb = self.enocde_sentences(sentences2)
            if self.similarity_func == "cosine":
                sent2_emb = normalize(sent2_emb, p=2, dim=1)
            logits = torch.mul(sent1_emb, sent2_emb).sum(dim=1)
        return self.loss(logits, labels)

    def enocde_sentences(self, sentences):
        output = self.bert_model(**sentences)
        return output.last_hidden_state.mean(dim=1)


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="siamese_cbow", 
        choices=["siamese_cbow", "multihead_attention", "transformer", "bert"],  help="specify which encoder to use")
    parser.add_argument("--bert_model", type=str, default="bert-base-chinese", help="specify which pretrained model to use")
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
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--batch_size", type=int, default=8, help="number of epochs")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--sim_func", type=str, default="dot", help="specify which similarity function to use")
    parser.add_argument("--device", type=str, default="cpu", help="specify which device to use")
    parser.add_argument("--use_tb", action="store_true", help="wether to use tensorboard")
    return parser


def train():
    parser = setup_argparser()
    args = parser.parse_args()

    if args.encoder != "bert":
        dataset = SentencePairDataset(args.data_file, args.min_tf, args.max_len)
        vocab_size = len(dataset.vocab) + 2

    if args.encoder == "siamese_cbow":
        model = SiameseCbowEncoder(vocab_size, args.emb_dims, args.max_len, pooling=args.pooling)
    elif args.encoder == "multihead_attention":
        model = MultiheadAttentionEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, pooling=args.pooling)
    elif args.encoder == "transformer":
        model = TransformerEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, args.num_layers, pooling=args.pooling)
    else:
        model = BertEncoder(args.bert_model)
        dataset = BertDataset(args.data_file, model.tokenizer, args.max_len)

    device = torch.device(args.device)
    model.to(device)
    model.train()

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    if args.encoder != "bert":     
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = None
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        num_train_steps = math.ceil(len(dataset) * args.epochs / args.batch_size)
        num_warmup_steps = int(args.warmup_proportion * num_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    if args.use_tb:
        tb_writer = SummaryWriter()

    global_step = 0
    log_interval = 50
    for _ in trange(args.epochs, desc="Epoch"):
        total_loss = 0
        pbar = tqdm(data_loader, desc="Interation")
        for step, batch in enumerate(pbar):
            if args.encoder != "bert":
                batch = [item.to(device) for item in batch]
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                sentences1 = {k.replace("sent1_", ""): v for k, v in batch.items() if k.startswith("sent1_")}
                sentences2 = {k.replace("sent2_", ""): v for k, v in batch.items() if k.startswith("sent2_")}
                labels = batch.get("label")
                batch = [sentences1, sentences2, labels]
            loss = model(*batch)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()
            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                pbar.set_postfix(loss=total_loss/log_interval)
                total_loss = 0
            if args.use_tb:
                tb_writer.add_scalar("loss", loss.item(), global_step)
            global_step += 1
    if args.encoder != "bert":
        model_path = os.path.join(args.save_dir, "encoder_model.pt")
        torch.save(model, model_path)
        with open(os.path.join(args.save_dir, "vocab.json"), "w", encoding="utf-8") as fo:
            json.dump(dataset.vocab, fo)
    else:
        model.bert_model.save_pretrained(args.save_dir)
        model.tokenizer.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, "args.json"), "w") as fo:
        args = {"encoder": args.encoder, "similarity_func": args.sim_func, 
                "max_length": args.max_len, "pooling": args.pooling}
        json.dump(args, fo)


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
    
    # model= BertEncoder("C:/code/models/chinese_base")
    # sentences1 = model.tokenizer(["你好吗", "你好美"], return_tensors="pt")
    # sentences2 = model.tokenizer(["你好呀", "你很美"], return_tensors="pt")
    # labels = torch.tensor([0, 1], dtype=torch.float)
    # print(model(sentences1, sentences2, labels))
    # print(model(sentences1))

    train()
