from semantic_matching.encoder import *
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from .dataset import BertDataset, TextGroupDataset
import argparse
import os.path
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import math
import json


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="siamese_cbow", 
        choices=["siamese_cbow", "multihead_attention", "transformer", "bert"],  help="specify which encoder to use")
    parser.add_argument("--bert_model", type=str, default="bert-base-chinese", help="specify which pretrained model to use")
    parser.add_argument("--data_file", type=str, help="path of the data file")
    parser.add_argument("--save_dir", type=str, help="path of the directory to save model")
    parser.add_argument("--with_negative", action="store_true", help="whether to train with hard negatives")
    parser.add_argument("--min_tf", type=int, default=3, help="minimum term frequence")
    parser.add_argument("--max_len", type=int, default=30, help="maximum number of tokens per sentence")
    parser.add_argument("--emb_dims", type=int, default=100, help="embedding dimensions")
    parser.add_argument("--num_heads", type=int, default=5, help="number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="number of transformer layers")
    parser.add_argument("--pooling", type=str, default="mean", choices=['mean', 'attention', 'full_connection', 'cls'], 
        help="type of pooling, incuding 'mean', 'attention', 'full_connection'")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--batch_size", type=int, default=8, help="number of epochs")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--t", type=float, default=0.05, help="the temperature hyperparameter")
    parser.add_argument("--device", type=str, default="cpu", help="specify which device to use")
    parser.add_argument("--use_tb", action="store_true", help="wether to use tensorboard")
    return parser


def train():
    parser = setup_argparser()
    args = parser.parse_args()

    if args.encoder != "bert":
        cache_path = os.path.join(os.path.dirname(args.data_file), ".sent_pairs")
        dataset = TextGroupDataset(args.data_file, args.min_tf, args.max_len, 
            with_negative=args.with_negative, cache_path=cache_path)
        vocab_size = dataset.vocab_size
    if args.encoder == "siamese_cbow":
        model = SiameseCbowEncoder(vocab_size, args.emb_dims, args.max_len, temperature=args.t, pooling=args.pooling)
    elif args.encoder == "multihead_attention":
        model = MultiheadAttentionEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, temperature=args.t, pooling=args.pooling)
    elif args.encoder == "transformer":
        model = TransformerEncoder(vocab_size, args.emb_dims, args.num_heads, args.max_len, args.num_layers, temperature=args.t, pooling=args.pooling)
    else:
        model = BertEncoder(args.bert_model, temperature=args.t, pooling=args.pooling)
        cache_path = os.path.join(os.path.dirname(args.data_file), ".bert")
        dataset = BertDataset(args.data_file, model.tokenizer, args.max_len, cache_path=cache_path)

    device = torch.device(args.device)
    model.to(device)
    model.train()

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    if args.encoder != "bert":     
        optimizer = AdamW(model.parameters(), lr=args.lr)
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
                sentences2 = sentences2 or None
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
        args = {"encoder": args.encoder, "temperature": args.t, 
                "max_length": args.max_len, "pooling": args.pooling}
        json.dump(args, fo)


if __name__ == "__main__":
    train()
