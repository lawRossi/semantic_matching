import os.path
import torch
from .encoder import *
import json


class EncoderWapper:
    def __init__(self, model_dir, tokenizer=None, device="cpu") -> None:
        with open(os.path.join(model_dir, "args.json")) as fi:
            self.args = json.load(fi)
        if self.args["encoder"] != "bert":
            model_path = os.path.join(model_dir, "encoder_model.pt")
            self.model = torch.load(model_path, map_location=device)
            with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fi:
                self.vocab = json.load(fi)
        else:
            self.model = BertEncoder(model_dir)
            self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def encode_sentences(self, sentences):
        max_length = self.args["max_length"]
        if self.args["encoder"] != "bert":
            input_ids = []
            for sentence in sentences:
                tokens = self.tokenizer(sentence)
                token_idxes = [self.vocab.get(token, len(self.vocab)+1) for token in tokens]
                token_idxes = token_idxes[:max_length] + [0] * (max_length - len(token_idxes))
                input_ids.append(token_idxes)
            intput_tensors = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            return self.model.enocde_sentences(intput_tensors).cpu().detach().numpy()
        else:
            sentences = self.model.tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            sentences = {k: v.to(self.device) for k, v in sentences.items()}
            return self.model.enocde_sentences(sentences).cpu().detach().numpy()


if __name__ == "__main__":
    import jieba

    encoder = EncoderWapper("output", jieba.lcut)
    print(encoder.encode_sentences(["我累了"]))
