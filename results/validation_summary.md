# Cleanup verification - 2026-10-08

Verified runtime: Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
PyTorch 2.14.0, Matplotlib 3.11.1. The packages were already installed in the
interpreter used for verification; a fresh virtual environment installation
was not performed.

- All eight Python modules imported successfully without running their main routines.
- The network, state equations, RK4 calculations, preprocessing and training
  loop preserve their original numerical logic.
- All ten saved checkpoint files match the bytes in the user's original backup.
- All ten full 4096-sample predicted trajectories are bit-for-bit identical to
  predictions collected before cleanup.
- Nine primary experiments were evaluated from their checkpoints. Their CSV
  references existing checkpoint, full-run plot and test plot files.
- The earlier extra `model_a.pt` run was evaluated separately.
- Model E: train RMSE 0.012396316 V; test RMSE 0.024515912 V.
- Data validation checked the exact raw-file excerpt, prescribed full-excerpt
  centering, stored means, sampling frequency and 3072/1024 split.
- RK4 step and rollout matched the analytical constant-acceleration solution.
- Training gradients through RK4 were finite and nonzero.
- A two-step Model E training smoke run completed with finite losses, saving
  only a temporary checkpoint outside this project. Full retraining was not run.
- Repeating the same training configuration saves a new numbered checkpoint
  without overwriting previous weights (updated VS Code workflow).
- Model E's regenerated full-run/test plots and the comparison chart were
  visually inspected; axis labels, legends and curves were readable.

The CSV scores describe the assignment's prescribed evaluation. Whole-excerpt
centering and test-based selection are explicit assignment requirements.
No independent holdout performance or additional optimizer sweep is claimed.
GitHub publication and the LaTeX report have not been performed.

## VS Code workflow update

- Training and evaluation settings are visible and commented at the top of each script.
- Command-line argument parsing was removed from these two scripts.
- The training optimization loop and all existing evaluation calculation functions
  are structurally identical to the pre-update functions.
- Two repeated two-step training smoke runs saved separate files with identical weights.
- Latest-checkpoint selection, explicit selection, all-model selection and the
  explanatory missing-file error were checked.
- A Model E evaluation completed after the update and retained test RMSE 0.024515912 V.
- Temporary training outputs stayed outside the project. The user's 100-step repeat
  configuration was preserved; no full training run was performed in this update.
