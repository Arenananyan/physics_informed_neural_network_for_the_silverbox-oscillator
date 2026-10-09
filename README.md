# Silverbox oscillator: neural acceleration model

Homework 1, Task 5 for PhotonicsAI: AI-Driven Science and Autonomous Laboratories.
The project learns the acceleration of the Silverbox oscillator with a PyTorch
network and integrates its state using classical fourth-order Runge-Kutta (RK4).

## Model and numerical method

The physical reference equation is

```text
y'' + a*y' + b*y + c*y^3 = g*u(t)
```

The implemented state equation is `y' = v`, `v' = alpha_theta(y, v, u)`.
The network learns acceleration directly; it does not fit explicit values of
`a`, `b`, `c`, or `g`, and the loss contains no separate physics-residual penalty.
The state equation and differentiable integrator provide the dynamical structure.

The lowest recorded test RMSE belongs to Model F repeat 01, with architecture
**3-256-256-1**. Both hidden layers use Tanh activations and the output is
linear. Inputs `[y, v, u]` are divided by `[0.1, 40.0, 0.05]`; the network
output is multiplied by `20000.0`. These fixed scales are identical across
the saved experiments. Model E (3-128-128-1) is retained for comparison.

Each RK4 step uses the measured input at both endpoints and their average at
the midpoint. Training uses randomly selected windows from the training record.
Each window starts from its measured output and a forward finite-difference
output derivative. The objective is mean squared output error through the
window. Optimization uses Adam and gradient clipping with maximum norm 1.0.
Model F uses a window of 128 samples, 3000 optimizer steps, and learning rate
0.0005. Repeat 01 used seed 42; the new independent repeat 02 used seed 43.
A step is one sampled window, not a full epoch.
The original CPU float32 calculation is preserved.

## Data and evaluation protocol

Place the public Silverbox file `SNLS80mV.mat` in `data/raw/`.
The existing dataset files and their original README files are preserved.
The source is the [Silverbox benchmark](https://www.nonlinearbenchmark.org/).

- Input: MATLAB channel `V1`; output: channel `V2`.
- Sampling frequency: 610.35 Hz; time step: `1 / 610.35` seconds.
- Python excerpt: `[42650:46746]`, containing 4096 samples.
- Subtract each signal's mean over the **whole excerpt**, as explicitly required
  by the homework, before splitting.
- Training: first 3072 centered samples; testing: last 1024.

Evaluation starts once at the beginning of the excerpt, using the measured
initial output and its forward finite-difference derivative. One continuous
4096-sample free run spans both segments. **The state is not reset at the test
boundary.** After initialization the rollout receives input voltage only;
measured output is used for scoring and plotting, not state correction.

The zero baseline is the RMSE of predicting zero on the centered test output.
The test plot uses the last 1024 predictions from this same free run.

## Verified results

The table shows only the run with the lowest test RMSE for each model A--F.
All 13 saved experiments, including the other settings and repeat runs, remain
in the detailed
[`results/tables/experiment_results.csv`](results/tables/experiment_results.csv).

| Model | Architecture | Window | Steps | LR | Train RMSE (V) | Test RMSE (V) |
|---|---|---:|---:|---:|---:|---:|
| A | 3-4-4-1 | 128 | 200 | 0.001 | 0.054946 | 0.061760 |
| B | 3-16-16-1 | 128 | 1000 | 0.001 | 0.044517 | 0.051304 |
| C | 3-32-32-1 | 128 | 1000 | 0.001 | 0.024891 | 0.030593 |
| D | 3-64-64-1 | 128 | 1000 | 0.0005 | 0.016129 | 0.026024 |
| E | 3-128-128-1 | 128 | 2000 | 0.0005 | 0.012396 | 0.024516 |
| **F** | **3-256-256-1** | **128** | **3000** | **0.0005** | **0.012311** | **0.013226** |

The zero baseline is **0.060492 V**. Model F repeat 01 has the lowest recorded
test RMSE: **0.013226 V** (13.2 mV), 78.1% below the zero baseline and 46.1%
below Model E's 0.024516 V. The independent F repeat 02 has a much larger test
RMSE of 0.046067 V, despite the same architecture and training budget. Thus a
single favorable seed should not be taken as typical performance. The second
E repeat reuses the same seed and reproduces its first result; it is not an
independent robustness check.

![Model F repeat 01 continuous free run](results/figures/model_f_repeat_01_w128_s3000_free_run.png)

![Model F repeat 01 test segment](results/figures/model_f_repeat_01_w128_s3000_test.png)

![Best run for each model A--F](results/figures/experiment_comparison.png)

### Interpretation and historical files

The assignment explicitly requests selecting the best test score. Accordingly,
this test record was used for model comparison; the selected score is not an
independent, untouched final test estimate. Full-excerpt centering also follows
the assignment rather than a general train-only preprocessing protocol.

Some comparisons change multiple settings at once: C to D changes width and
learning rate; D to E changes width and training steps; E to F changes width
and training steps. The measured improvement cannot be attributed to width
alone. The current training script uses Adam. No additional optimizer sweep is
claimed.

`results/checkpoints/model_a.pt` is an earlier extra run, distinct from
`model_a_w16.pt`. Its recomputed train/test RMSE is approximately
9.228715/18.489641 V. It and its original figure are preserved, but it is excluded
from the comparison table. Its separate results can be regenerated with
the single-checkpoint evaluation command below. Original figures are retained;
new evaluation figures are named after their actual checkpoint stems.

## Project structure

```text
README.md
requirements.txt
.gitignore
scr/                         # Existing folder name preserved
    data_handler.py          # Excerpt, prescribed centering, metadata
    model.py                 # Scaled acceleration network
    integrator.py            # Differentiable RK4 step and rollout
    train.py                 # Random-window training and checkpoint saving
    evaluate.py              # Continuous free-run scoring, figures, CSV
    visualize.py             # Centered input/output visualization
    compare_experiments.py   # Comparison chart from the verified CSV
    validate.py              # Preprocessing and integrator checks
data/
    raw/                     # Original dataset and source notes
    processed/silverbox_excerpt.npz
results/
    checkpoints/             # Saved experiments; retained for comparison
    figures/                 # Original and regenerated plots
    tables/                  # Recomputed experiment metrics
```

## Setup

Verification used Python **3.14.6** and the exact package versions in
`requirements.txt`. To create an isolated environment with that Python version:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All commands below run from the repository root. No absolute user-specific paths
are required. Model loading uses `torch.load(..., weights_only=True)` on CPU.

## Run in VS Code

Open the project folder in VS Code and select the Python interpreter containing
the packages from `requirements.txt`. Open a script and press **Run Python File**.
All project paths are resolved from the script location, so Run also works when
the terminal is currently in another directory.

1. Run `scr/data_handler.py` to prepare the prescribed excerpt if needed.
2. Run `scr/visualize.py` to view the centered signals.
3. Run `scr/train.py` when you want a new training experiment.
4. Run `scr/evaluate.py` to score a saved model and save its figures and CSV.

### Training settings

Edit the commented **Training configuration** block at the top of `train.py`.
It contains `model_name`, `hidden_size`, `num_steps`, `window_size`,
`learning_rate`, and `random_seed`. The current configuration runs an independent
Model F repeat with 256 hidden neurons per layer, 3000 steps, window 128,
learning rate 0.0005, and seed 43. Change the model name and seed when starting
a distinct repeat experiment.

If the checkpoint filename already exists, the script adds `_run02`, `_run03`,
etc. automatically. Existing weights are never overwritten. The output path is
printed before training and again when saving. A fixed seed improves
reproducibility, but retraining on another runtime or platform is not guaranteed
to reproduce every saved weight exactly.

### Evaluation settings

Edit the commented **Evaluation configuration** block at the top of `evaluate.py`:

```python
checkpoint_name = None  # Use the most recently saved checkpoint.
evaluate_all = False   # Evaluate one selected model.
show_plots = True      # Display plots as well as saving them.
```

`None` chooses the checkpoint with the newest file modification time and prints
its name before evaluation. To evaluate the lowest-scoring Model F run, use:

```python
checkpoint_name = "model_f_repeat_01_w128_s3000.pt"
```

To rebuild the comparison CSV for all saved experiments, set `evaluate_all = True`.
This includes new repeat runs and excludes only the earlier extra `model_a.pt`.
Set it back to `False` for single-model evaluation. A single-model run writes its
own CSV and never replaces the complete comparison table.

Run `scr/compare_experiments.py` to plot the comparison table, and
`scr/validate.py` to check preprocessing, RK4 and training gradients.
Set `show_plots = False` for evaluation without opening windows.

The old command-line configuration flags in `train.py` and `evaluate.py` have
been removed in favor of the visible settings above. Both can also be run from
the repository root with `python scr/train.py` and `python scr/evaluate.py`.

## Cleanup verification

See [`results/validation_summary.md`](results/validation_summary.md) for the
checks actually run during cleanup.

## GitHub preparation

The code, processed excerpt, checkpoints, plots and result tables are retained.
`.gitignore` excludes virtual environments, Python caches, macOS metadata,
editor settings, temporary work and local `.env` files. The project is published
at [GitHub](https://github.com/Arenananyan/physics_informed_neural_network_for_the_silverbox-oscillator).
