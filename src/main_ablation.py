# main_ablation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
import numpy as np
import argparse  # New import for command-line arguments
import os  # New import for creating directories

from datasets import load_from_disk
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------------
# 1. Overall Model Architecture (Encoder-Decoder)
# ----------------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """
    Standard Encoder-Decoder architecture.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Process masked source and target sequences."
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(memory, src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ----------------------------------------------------------------------------------
# 2. Encoder Part
# ----------------------------------------------------------------------------------

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers."

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (torch.sqrt(std ** 2 + self.eps)) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward."

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ----------------------------------------------------------------------------------
# 3. Decoder Part
# ----------------------------------------------------------------------------------

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward."

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


# ----------------------------------------------------------------------------------
# 4. Attention Mechanism
# ----------------------------------------------------------------------------------

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ----------------------------------------------------------------------------------
# 5. Other Core Components (FFN, Embeddings, Positional Encoding)
# ----------------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# 6. Model Building Function (MODIFIED for ablation)
# ----------------------------------------------------------------------------------

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, use_pe=True):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # Create embedding layers, conditionally adding positional encoding
    src_embed_layers = [Embeddings(d_model, src_vocab)]
    if use_pe:
        src_embed_layers.append(c(position))

    tgt_embed_layers = [Embeddings(d_model, tgt_vocab)]
    if use_pe:
        tgt_embed_layers.append(c(position))

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(*src_embed_layers),
        nn.Sequential(*tgt_embed_layers),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------
# 7. Data Loading and Preprocessing (Unchanged)
# ----------------------------------------------------------------------------------

# --- (此处省略你已有的所有数据处理代码，从 tokenize_en 到 collate_fn) --->
# --- Step 1: Define Tokenizers ---
def tokenize_en(text):
    """English tokenizer"""
    return text.split()


def tokenize_kn(text):
    """
    Kannada tokenizer.
    This is a basic space-based tokenizer. For better performance,
    consider using a dedicated Kannada NLP library.
    """
    return text.split()


# --- Step 2: Define Data Loading Function (MODIFIED) ---
def load_data(dataset_dir):
    """
    Load data from a Hugging Face datasets format directory.
    MODIFIED to load 'en' and 'kn' fields.
    """
    src_data, tgt_data = [], []
    print(f"Loading dataset from disk: {dataset_dir}...")

    try:
        dataset = load_from_disk(dataset_dir)
    except FileNotFoundError:
        print(f"Error: Dataset directory not found at '{dataset_dir}'.")
        return [], []

    for item in tqdm(dataset):
        translation_pair = item.get('translation', {})
        en_text = translation_pair.get('en')
        # <--- MODIFICATION: Changed 'zh' to 'kn' --->
        kn_text = translation_pair.get('kn')
        if en_text and kn_text:
            src_data.append(en_text)
            tgt_data.append(kn_text)

    print("Data loading complete.")

    # --- Added Check ---
    if not src_data or not tgt_data:
        raise ValueError(f"No valid data loaded from {dataset_dir}. "
                         f"Please check the dataset path and format. "
                         f"The code expects items with a structure like: "
                         f"{{'translation': {{'en': 'english text', 'kn': 'kannada text'}}}}")

    return src_data, tgt_data


# --- Step 3: Build Vocabulary ---
from torchtext.vocab import build_vocab_from_iterator


def build_vocab(texts, tokenizer, special_symbols, min_freq=2):
    def yield_tokens(texts_iter):
        for text in tqdm(texts_iter, desc="Building vocab"):
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(texts),
                                      min_freq=min_freq,
                                      specials=special_symbols,
                                      special_first=True)

    vocab.set_default_index(vocab['<unk>'])
    return vocab


# --- Step 4: Define Data Processing and Batching ---
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']


def sequential_transform(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


# ----------------------------------------------------------------------------------
# 8. Main Function (MODIFIED for ablation)
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    # --- New: Command-line Argument Parser ---
    parser = argparse.ArgumentParser(description='Transformer Ablation Study')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for saving files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--N', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--h', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--no_pe', action='store_true', help='Disable positional encoding')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    set_seed(args.seed)
    # --- Configuration from args ---
    TRAIN_FILE_PATH = "data/train"
    VAL_FILE_PATH = "data/validation"

    # Dynamic model and plot paths
    os.makedirs("models", exist_ok=True)  # Create a new folder for models
    os.makedirs("results", exist_ok=True)
    MODEL_PATH = f"models/transformer_{args.exp_name}.pth"
    PLOT_PATH = f"results/loss_curve_{args.exp_name}.png"

    BATCH_SIZE = args.batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Experiment: {args.exp_name} ---")
    print(f"Using device: {DEVICE}")
    print(f"Parameters: N={args.N}, h={args.h}, d_model={args.d_model}, d_ff={args.d_ff}, PE={not args.no_pe}")

    # --- Data Loading and Preprocessing ---
    train_src_texts, train_tgt_texts = load_data(TRAIN_FILE_PATH)
    val_src_texts, val_tgt_texts = load_data(VAL_FILE_PATH)

    print("Building English vocab...")
    src_vocab = build_vocab(train_src_texts, tokenize_en, special_symbols)
    print("Building Kannada vocab...")
    tgt_vocab = build_vocab(train_tgt_texts, tokenize_kn, special_symbols)

    SRC_VOCAB_SIZE = len(src_vocab.get_itos())
    TGT_VOCAB_SIZE = len(tgt_vocab.get_itos())

    src_text_transform = sequential_transform(tokenize_en, src_vocab, tensor_transform)
    tgt_text_transform = sequential_transform(tokenize_kn, tgt_vocab, tensor_transform)


    # --- Create DataLoader ---
    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, src_texts, tgt_texts):
            self.src_texts = src_texts
            self.tgt_texts = tgt_texts

        def __len__(self):
            return len(self.src_texts)

        def __getitem__(self, idx):
            return src_text_transform(self.src_texts[idx]), tgt_text_transform(self.tgt_texts[idx])


    train_dataset = TranslationDataset(train_src_texts, train_tgt_texts)
    val_dataset = TranslationDataset(val_src_texts, val_tgt_texts)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # --- Model, Loss Function, and Optimizer (using args) ---
    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                       N=args.N, d_model=args.d_model, d_ff=args.d_ff,
                       h=args.h, dropout=args.dropout, use_pe=(not args.no_pe))
    model = model.to(DEVICE)

    print(f'The model for experiment "{args.exp_name}" has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    train_losses = []
    val_losses = []

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = (src != PAD_IDX).unsqueeze(-2)
            tgt_mask = (tgt_input != PAD_IDX).unsqueeze(-2) & subsequent_mask(tgt_input.size(1)).to(DEVICE)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            optimizer.zero_grad()

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Evaluation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                src_mask = (src != PAD_IDX).unsqueeze(-2)
                tgt_mask = (tgt_input != PAD_IDX).unsqueeze(-2) & subsequent_mask(tgt_input.size(1)).to(DEVICE)

                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        print(
            f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Val loss: {avg_val_loss:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- Save Model ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # --- Visualize Training Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss Curve\n({args.exp_name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"Loss curve plot saved to {PLOT_PATH}")
    print(f"--- Finished Experiment: {args.exp_name} ---")