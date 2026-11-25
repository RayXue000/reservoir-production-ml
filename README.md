# Reservoir Production ML — Selected Core

An LSTM sequence-model excerpt from Rui Xue's petroleum machine-learning
software project. 气藏产能机器学习软件中的时序模型核心节选。

## Source and contents

- `model.py`: the `SimpleLSTM` class extracted from cell 4 of the original
  project's `ML.ipynb`; the model architecture is unchanged.
- `sequence.py`: the next-step sliding-window construction refactored into
  a data-independent function with input checks.
- `example.py`: a new synthetic run with a chronological train/test boundary,
  a few optimization steps and an inference shape check.

The original notebook's GUI state, exploratory outputs and full-series
preprocessing are not carried over. If adding normalization, fit it on the
training period only. Splits and preprocessing must be designed for the
intended forecasting horizon before evaluating any real dataset.

## Run

Python 3.10+:
```sh
python -m pip install -r requirements.txt
python example.py
```

This small release shows the core recurrent architecture. It is not the
complete multi-model desktop application, an optimization study, or a
reproduction of the project's published/displayed evaluation figures.
No real well identifiers, production records, input spreadsheets, trained
weights, GUI resources or third-party tutorial notebooks are distributed.

[Academic homepage](https://rayxue000.github.io/)
