# tf1-chess-ai-policy-value-mcts

An archived **TensorFlow 1.x** chess AI prototype using **policy/value networks + MCTS (PUCT-style)**, based on ideas from the AlphaGo (2016) paper.

This repository is intended as a **portfolio / learning snapshot** rather than a polished, fully reproducible engine. Some training data, checkpoints, and glue modules referenced by the code are not included.

## Table of contents

- [What this project is (and isn’t)](#what-this-project-is-and-isnt)
- [Background (AlphaGo 2016 → Chess)](#background-alphago-2016--chess)
- [Repository structure](#repository-structure)
- [Architecture](#architecture)
- [Representations](#representations)
- [Data pipeline (PGN/FEN)](#data-pipeline-pgnfen)
- [How to run (best-effort)](#how-to-run-best-effort)
- [Known issues / limitations](#known-issues--limitations-snapshot)
- [Results (plots)](#results-plots)
- [Reference](#reference)

## What this project is (and isn’t)

What it is:

- **Policy network (supervised):** predicts a move distribution from a board position (`4096 = 64×64` from-to encoding).
- **Value network (supervised experiments):** predicts outcome/value from a board position (multiple variants explored).
- **MCTS skeleton (PUCT-like):** node selection combines exploitation + exploration (`Q + U`) and is designed to expand children using the policy network priors.

What it is not (in this snapshot):

- **Reinforcement learning / self-play training:** there is no loop that updates networks from rewards/outcomes (no policy gradient, no AlphaZero-style self-play).
- **A complete AlphaGo/AlphaZero implementation:** this is a prototype to learn the components, not a production-strength system.
- **Fully reproducible training artifacts:** datasets and checkpoints referenced by scripts are not tracked here.

## Background (AlphaGo 2016 → Chess)

The goal was to take the *high-level algorithmic pattern* from AlphaGo (2016) and apply it to chess:

- a **policy network** proposes promising moves,
- a **value network** estimates how good a position is,
- **MCTS** uses the networks’ outputs to guide search.

In practice, this repo focuses on getting the pipeline pieces working and experimenting with representations/training. The training scripts here are primarily **supervised learning**.

## Repository structure

- `ChessGamePlay.py`: CLI chess game (human vs AI, AI vs AI) using `python-chess`.
- `ChessAI.py`: small wrapper that delegates move selection to the search module.
- `Montecarlo.py`: MCTS search loop (selection/expansion/simulation/backprop). Imports `Tree` (missing in this snapshot).
- `Node.py`: node statistics + PUCT-like scoring (`calc_Q`, `calc_u`).
- `Board2Array.py`: feature-plane encoder for board states.
- `OneHotEncoding.py`: 4096 move encoding + symmetry helpers.
- `GetMovesAndScores.py`: TensorFlow session for inference-style “move scoring” (policy network restore + legal-move filtering).
- `PolicyNetwork/`: policy network training code + plots.
- `MakingValueNetwork/`: value network training code + value-data generation utilities.
- `Preprocessing/`: PGN/FEN preprocessing experiments (building datasets).
- `Sunfish/`, `PlayGameWithSunfish.py`: experiments playing against the Sunfish engine.

## Architecture

Intended end-to-end flow:

1. Convert a `python-chess` board to an `8×8×C` tensor of feature planes.
2. Use a policy network to compute priors over 4096 from-to moves.
3. Run MCTS, selecting children via a PUCT-like score and expanding using policy priors.
4. Choose the move by most visits (or similar).

### Policy network

- Training script: `PolicyNetwork/LearningPolicyNetwork.py`
- Input: `8×8×36` (in the `PolicyNetwork` variant of the feature encoder)
- Output: `4096` logits (from-to move classes)
- Loss: softmax cross-entropy against one-hot 4096 targets
- Implementation: TensorFlow 1.x (`tf.Session`, placeholders, `tf.contrib.layers.xavier_initializer`)

### Value network (experiments)

Several variants exist in this snapshot:

- `learning_ValueNetwork_Using_Softmax.py`: 3-class softmax for `{white win, draw, black win}`
- `MakingValueNetwork/Learning_ValueNetwork.py`: scalar output (tanh) value-like prediction

### Search (MCTS / PUCT-like)

- Search loop: `Montecarlo.py`
- Node stats + scoring: `Node.py`
  - `Q`: empirical result statistics plus optional value term
  - `U`: `Cpuct * policy_prior * sqrt(parent_visits) / (1 + visits)`

## Representations

### Move representation: 4096 from-to head

Moves are encoded as a `64×64` grid (from-square × to-square) flattened to 4096 classes.

- Implementation: `OneHotEncoding.py`
- Note: this representation is a simplification and does **not** explicitly model special moves as separate action types (e.g. promotions). Some promotions can be ambiguous if only `from-to` is used.

### Board representation: 8×8×C planes

Boards are converted into multiple feature planes (piece maps + tactical/metadata planes). The exact number of channels varies across scripts/experiments (e.g. 33 vs 36).

One commonly used feature set in this repo includes planes such as:

- piece/color occupancy
- per-piece planes (K/Q/R/B/N/P for both sides)
- sliding-piece attack maps (via `python-chess` attack generation)
- pinned piece map
- attacked squares map
- check indicator
- castling rights indicators
- (optional) turn planes

See `Board2Array.py` and `PolicyNetwork/Board2Array.py` for the concrete plane definitions used by each script.

### Symmetry for black

Some preprocessing and move mapping normalize black-to-move positions by mirroring them into a white perspective.
See `OneHotEncoding.py` and parts of `Preprocessing/` for symmetry helpers.

## Data pipeline (PGN/FEN)

There are multiple data-prep experiments:

- `Preprocessing/` contains PGN reading and feature/label generation attempts.
- Several training scripts expect a line-based dataset with entries similar to:

  - `FEN:uci_move:result`

Dataset filenames/locations are environment-specific and not included in this snapshot.

## How to run (best-effort)

This repo targets TensorFlow 1.x and `python-chess`. It may not run as-is in a modern TF2 environment.

### CLI chess game

- Entry point: `ChessGamePlay.py`

### Training

- Policy network training: `PolicyNetwork/LearningPolicyNetwork.py`
- Value network experiments: `MakingValueNetwork/*`, `learning_ValueNetwork_Using_Softmax.py`

Most training scripts assume local data and may require path edits.

## Known issues / limitations (snapshot)

- The MCTS code imports a `Tree` module (e.g. `import Tree as TR`), but `Tree.py` is not present in this snapshot, so end-to-end MCTS gameplay may require reconstruction.
- Many scripts assume local files (e.g. `FenData/allFen.txt`, `KingBase/`, checkpoint folders) that are not tracked here.
- TensorFlow 1.x (`tf.Session`, `tf.contrib`) code will not run unmodified on TF2.
- Some scripts contain hard-coded paths from the original development machine.

## Results (plots)

- `PolicyNetwork/epoch5_costNaccuracy.PNG`
- `PolicyNetwork/epoch5_costNaccuracyWithoutOutlier.PNG`

## Reference

- David Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016).
