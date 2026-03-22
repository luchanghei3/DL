# Assignment 2 Questions 1-2

This repository now includes a runnable Chinese-to-English neural machine translation pipeline for the first two questions of Assignment 2.

## Files

- `assignment2_seq2seq_zh_en.py`: end-to-end data processing, train/test split, seq2seq-with-attention model, evaluation, and report generation.
- `bpe.py`: GPT-2 BPE helper reused as an optional English subword tokenizer.
- `question1_2_report.md`: generated after a training run.

## Dataset

1. Download `cmn-eng.zip` from the assignment link.
2. Extract the parallel corpus file, typically `cmn.txt`.
3. Run the training script from the repository root.

## Example command

```bash
python assignment2_seq2seq_zh_en.py \
  --data-path /path/to/cmn.txt \
  --tokenizer word \
  --epochs 10 \
  --batch-size 64 \
  --report-path question1_2_report.md
```

If the GPT-2 BPE assets are available in your environment, you can also use:

```bash
python assignment2_seq2seq_zh_en.py --data-path /path/to/cmn.txt --tokenizer bpe
```

## Output

The script prints training metrics each epoch and writes a Markdown report containing:

- dataset statistics for Question 1,
- training history for Question 2,
- at least 10 example predictions,
- attention tensor shapes for downstream visualization work.
