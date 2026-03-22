# Assignment 2 Questions 1-2

This repository now includes a runnable Chinese-to-English neural machine translation pipeline for the first two questions of Assignment 2, with Chinese-aware tokenization as the core preprocessing focus.

## Files

- `assignment2_seq2seq_zh_en.py`: end-to-end data processing, Chinese-source tokenization (BPE or character fallback), train/test split, seq2seq-with-attention model, evaluation, and report generation.
- `bpe.py`: GPT-2 BPE helper reused as the Chinese-source subword tokenizer backend when BPE mode is enabled.
- `question1_2_report.md`: generated after a training run.

## Dataset

1. Download `cmn-eng.zip` from the assignment link.
2. Extract the parallel corpus file, typically `cmn.txt`.
3. Run the training script from the repository root.

## Example command

```bash
python assignment2_seq2seq_zh_en.py \
  --data-path /path/to/cmn.txt \
  --source-tokenizer bpe \
  --target-tokenizer word \
  --epochs 10 \
  --batch-size 64 \
  --report-path question1_2_report.md
```

If the GPT-2 BPE assets are unavailable, the script will automatically fall back to a Chinese character tokenizer. You can also force that mode explicitly:

```bash
python assignment2_seq2seq_zh_en.py --data-path /path/to/cmn.txt --source-tokenizer char
```

## Output

The script prints training metrics each epoch and writes a Markdown report containing:

- dataset statistics for Question 1,
- the Chinese tokenization strategy actually used for the source side,
- training history for Question 2,
- at least 10 example predictions,
- attention tensor shapes for downstream visualization work.
