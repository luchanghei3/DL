from __future__ import annotations

import argparse
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bpe import get_encoder

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_english(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"([.!?,';:()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_chinese_characters(text: str) -> List[str]:
    text = text.strip().replace(" ", "")
    return [ch for ch in text if ch.strip()]


def tokenize_english_words(text: str) -> List[str]:
    return normalize_english(text).split()


class GPT2BPETokenizerAdapter:
    """GPT-2 BPE adapter primarily used for Chinese-source tokenization.

    The assignment explicitly highlights that Chinese characters cannot be handled as
    ASCII codes, so this adapter exposes subword tokens for Chinese sentences.
    It can also be reused for English when needed.
    """

    def __init__(self) -> None:
        self.encoder = get_encoder()

    def __call__(self, text: str) -> List[str]:
        normalized_text = text.strip()
        token_ids = self.encoder.encode(normalized_text)
        return [self.encoder.decoder[token_id] for token_id in token_ids]


@dataclass
class Example:
    source_text: str
    target_text: str
    source_tokens: List[str]
    target_tokens: List[str]


class Vocabulary:
    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.idx_to_token = list(SPECIAL_TOKENS)
        self.frequencies: dict[str, int] = {}

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.frequencies[token] = self.frequencies.get(token, 0) + 1

    def build(self) -> None:
        for token, freq in sorted(self.frequencies.items(), key=lambda item: (-item[1], item[0])):
            if freq < self.min_freq or token in self.token_to_idx:
                continue
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def encode(self, tokens: Sequence[str], add_sos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_sos:
            ids.append(SOS_IDX)
        ids.extend(self.token_to_idx.get(token, UNK_IDX) for token in tokens)
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, token_ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        tokens = []
        for idx in token_ids:
            if stop_at_eos and idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            tokens.append(self.idx_to_token[idx])
        return tokens

    def __len__(self) -> int:
        return len(self.idx_to_token)


class TranslationDataset(Dataset):
    def __init__(self, examples: Sequence[Example], source_vocab: Vocabulary, target_vocab: Vocabulary) -> None:
        self.examples = list(examples)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        source_ids = self.source_vocab.encode(example.source_tokens)
        target_ids = self.target_vocab.encode(example.target_tokens)
        return torch.tensor(source_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def pad_batch(sequences: Sequence[torch.Tensor]) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    batch = torch.full((len(sequences), max_len), PAD_IDX, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : seq.size(0)] = seq
    return batch


def make_collate_fn() -> Callable[[Sequence[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]]:
    def collate_fn(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        source_batch, target_batch = zip(*batch)
        return pad_batch(source_batch), pad_batch(target_batch)

    return collate_fn


def read_parallel_corpus(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        english, chinese = parts[0].strip(), parts[1].strip()
        if english and chinese and contains_chinese(chinese):
            pairs.append((chinese, english))
    if not pairs:
        raise ValueError(f"No Chinese-English sentence pairs found in {path}")
    return pairs


def build_examples(
    pairs: Sequence[tuple[str, str]],
    source_tokenizer: Callable[[str], List[str]],
    target_tokenizer: Callable[[str], List[str]],
    max_source_len: int | None = None,
    max_target_len: int | None = None,
) -> list[Example]:
    examples = []
    for source_text, target_text in pairs:
        source_tokens = source_tokenizer(source_text)
        target_tokens = target_tokenizer(target_text)
        if not source_tokens or not target_tokens:
            continue
        if max_source_len and len(source_tokens) > max_source_len:
            continue
        if max_target_len and len(target_tokens) > max_target_len:
            continue
        examples.append(Example(source_text, target_text, source_tokens, target_tokens))
    return examples


def split_examples(examples: Sequence[Example], train_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_ids))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.score_proj = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_proj(torch.tanh(self.query_proj(query) + self.key_proj(keys))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward_step(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(decoder_input))
        query = decoder_hidden.transpose(0, 1)
        context, weights = self.attention(query, encoder_outputs, source_mask)
        gru_input = torch.cat([embedded, context], dim=-1)
        output, decoder_hidden = self.gru(gru_input, decoder_hidden)
        logits = self.output(output)
        return logits, decoder_hidden, weights


class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, embed_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = Encoder(source_vocab_size, embed_dim, hidden_size, dropout)
        self.decoder = AttentionDecoder(target_vocab_size, embed_dim, hidden_size, dropout)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor | None = None, teacher_forcing_ratio: float = 1.0):
        encoder_outputs, encoder_hidden = self.encoder(source_ids)
        source_mask = source_ids.ne(PAD_IDX)
        batch_size = source_ids.size(0)
        max_steps = target_ids.size(1) if target_ids is not None else 50

        decoder_input = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=source_ids.device)
        decoder_hidden = encoder_hidden
        logits_steps = []
        attention_steps = []

        for step in range(max_steps - 1):
            step_logits, decoder_hidden, step_attention = self.decoder.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, source_mask
            )
            logits_steps.append(step_logits)
            attention_steps.append(step_attention.unsqueeze(1))

            use_teacher = target_ids is not None and random.random() < teacher_forcing_ratio
            if use_teacher:
                decoder_input = target_ids[:, step + 1].unsqueeze(1)
            else:
                decoder_input = step_logits.argmax(dim=-1)

        logits = torch.cat(logits_steps, dim=1)
        attentions = torch.cat(attention_steps, dim=1)
        return logits, attentions


@torch.no_grad()
def greedy_decode(model: Seq2SeqAttentionModel, source_ids: torch.Tensor, max_steps: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, attentions = model(source_ids, target_ids=None, teacher_forcing_ratio=0.0)
    predictions = logits.argmax(dim=-1)
    return predictions, attentions


def ids_to_sentence(ids: Sequence[int], vocab: Vocabulary, join_without_space: bool = False) -> str:
    tokens = vocab.decode(ids)
    return "".join(tokens) if join_without_space else " ".join(tokens)


@torch.no_grad()
def evaluate(model: Seq2SeqAttentionModel, data_loader: DataLoader, source_vocab: Vocabulary, target_vocab: Vocabulary, device: torch.device) -> dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_tokens = 0
    exact_match = 0
    example_rows = []

    for source_ids, target_ids in data_loader:
        source_ids = source_ids.to(device)
        target_ids = target_ids.to(device)
        logits, attentions = model(source_ids, target_ids=target_ids, teacher_forcing_ratio=0.0)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))

        non_pad = target_ids[:, 1:].ne(PAD_IDX)
        total_loss += loss.item() * non_pad.sum().item()
        total_tokens += non_pad.sum().item()

        predictions = logits.argmax(dim=-1)
        for batch_idx in range(source_ids.size(0)):
            pred_sentence = ids_to_sentence(predictions[batch_idx].tolist(), target_vocab)
            gold_sentence = ids_to_sentence(target_ids[batch_idx].tolist(), target_vocab)
            if pred_sentence == gold_sentence:
                exact_match += 1
            if len(example_rows) < 10:
                example_rows.append(
                    {
                        "source": ids_to_sentence(source_ids[batch_idx].tolist(), source_vocab, join_without_space=True),
                        "prediction": pred_sentence,
                        "target": gold_sentence,
                        "attention_shape": tuple(attentions[batch_idx].shape),
                    }
                )

    return {
        "loss": total_loss / max(total_tokens, 1),
        "perplexity": math.exp(total_loss / max(total_tokens, 1)),
        "exact_match": exact_match / max(len(data_loader.dataset), 1),
        "examples": example_rows,
    }


def render_examples(examples: Sequence[dict[str, object]]) -> str:
    lines = ["| # | Chinese Input | Prediction | Ground Truth | Attention Shape |", "|---|---|---|---|---|"]
    for idx, row in enumerate(examples, start=1):
        lines.append(
            f"| {idx} | {row['source']} | {row['prediction']} | {row['target']} | {row['attention_shape']} |"
        )
    return "\n".join(lines)


def train_one_epoch(
    model: Seq2SeqAttentionModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for source_ids, target_ids in data_loader:
        source_ids = source_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits, _ = model(source_ids, target_ids=target_ids, teacher_forcing_ratio=teacher_forcing_ratio)
        target = target_ids[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        non_pad = target.ne(PAD_IDX)
        total_loss += loss.item() * non_pad.sum().item()
        total_tokens += non_pad.sum().item()

    return total_loss / max(total_tokens, 1)


def choose_source_tokenizer(mode: str) -> Callable[[str], List[str]]:
    if mode == "char":
        return tokenize_chinese_characters
    if mode == "bpe":
        try:
            return GPT2BPETokenizerAdapter()
        except Exception as exc:
            print(f"Falling back to Chinese character tokenizer because GPT-2 BPE setup failed: {exc}")
            return tokenize_chinese_characters
    raise ValueError(f"Unsupported source tokenizer mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chinese-to-English seq2seq with attention for Assignment 2 questions 1-2, using Chinese-aware tokenization.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the extracted cmn.txt-style parallel corpus.")
    parser.add_argument("--source-tokenizer", choices=["bpe", "char"], default="bpe", help="Chinese source tokenization strategy.")
    parser.add_argument("--target-tokenizer", choices=["word"], default="word", help="English target tokenization strategy.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--max-source-len", type=int, default=50)
    parser.add_argument("--max-target-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit on the number of examples.")
    parser.add_argument("--report-path", type=Path, default=Path("question1_2_report.md"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tokenizer = choose_source_tokenizer(args.source_tokenizer)
    target_tokenizer = tokenize_english_words
    pairs = read_parallel_corpus(args.data_path)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    examples = build_examples(pairs, source_tokenizer, target_tokenizer, args.max_source_len, args.max_target_len)
    train_examples, test_examples = split_examples(examples, args.train_ratio, args.seed)

    source_vocab = Vocabulary()
    target_vocab = Vocabulary()
    for example in train_examples:
        source_vocab.add_tokens(example.source_tokens)
        target_vocab.add_tokens(example.target_tokens)
    source_vocab.build()
    target_vocab.build()

    train_dataset = TranslationDataset(train_examples, source_vocab, target_vocab)
    test_dataset = TranslationDataset(test_examples, source_vocab, target_vocab)
    collate_fn = make_collate_fn()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqAttentionModel(len(source_vocab), len(target_vocab), args.embed_dim, args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing)
        metrics = evaluate(model, test_loader, source_vocab, target_vocab, device)
        history.append((epoch, train_loss, metrics["loss"], metrics["perplexity"], metrics["exact_match"]))
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | test_loss={metrics['loss']:.4f} | "
            f"test_ppl={metrics['perplexity']:.2f} | exact_match={metrics['exact_match']:.2%}"
        )

    elapsed = time.time() - start_time
    final_metrics = evaluate(model, test_loader, source_vocab, target_vocab, device)

    report_lines = [
        "# Assignment 2 - Questions 1 & 2",
        "",
        "## Question 1: Dataset Loading and Processing",
        f"- Dataset path: `{args.data_path}`",
        f"- Total usable pairs: {len(examples)}",
        f"- Training pairs: {len(train_examples)}",
        f"- Test pairs: {len(test_examples)}",
        f"- Chinese tokenizer: `{args.source_tokenizer}`",
        f"- English tokenizer: `{args.target_tokenizer}`",
        f"- Chinese vocabulary size: {len(source_vocab)}",
        f"- English vocabulary size: {len(target_vocab)}",
        "",
        "## Question 2: Seq2Seq with Attention",
        f"- Device: `{device}`",
        f"- Epochs: {args.epochs}",
        f"- Embedding dimension: {args.embed_dim}",
        f"- Hidden size: {args.hidden_size}",
        f"- Training time (seconds): {elapsed:.2f}",
        "",
        "## Training History",
        "",
        "| Epoch | Train Loss | Test Loss | Test Perplexity | Exact Match |",
        "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {epoch} | {train_loss:.4f} | {test_loss:.4f} | {ppl:.2f} | {exact_match:.2%} |"
        for epoch, train_loss, test_loss, ppl, exact_match in history
    )
    report_lines.extend(
        [
            "",
            "## Example Predictions",
            "",
            render_examples(final_metrics["examples"]),
            "",
            "## Conclusion",
            "",
            "This implementation completes the first two assignment requirements by "
            "(1) loading and preprocessing the full Chinese-English parallel corpus with a reproducible train/test split, and "
            "(2) training a Bahdanau-attention seq2seq model in PyTorch while recording quantitative evaluation metrics.",
        ]
    )
    args.report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report to {args.report_path}")


if __name__ == "__main__":
    main()
