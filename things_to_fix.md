# Things to fix

## For Matteo?

Issues found while using the `prpl-group/agentic-gentamp-runs` dataset (revision `7dec3d08`) with this repo.

- [ ] **`opus5/rovers/seed444`: the history is recoverable.** The only corrupt object is blob `974849ff`, which is `feats.pkl`, a pickled probe dump added at commit 3. `approach.py` never reads it at any commit. `git archive <sha>` fails for the 13 commits whose tree contains it, but exporting each commit without that file (`git archive <sha> -- <all paths except feats.pkl>`) recovers the exact `approach.py` for all 16 commits. So the missing per-commit re-evaluation may be doable. The other 979 restored histories pass `git fsck`.

- [ ] **The results viewer hides 26 dataset runs as shipped.** Checked against all 1,195 archives:
  - 16 runs record `approach.load_dir` in `.hydra/overrides.yaml` (re-scored after generation), so the viewer skips them as re-evaluation passes (`experiments/results_viewer.py:363`): 8 `opus55-source`, 5 `opus5` (all `dynamo3d`), and 3 `opus5-source`.
  - 10 runs have results only in `rescore/results.json`, with no top-level `results.json`: all `dynpusht2d` runs in `opus5` and `opus5-source`.

  Adding an `env_variant.json` marker (e.g. `{"approach": "agentic"}`) to those archives, and a top-level `results.json` for the `rescore/` ones, would make them show. Alternatively, the viewer could get an option for this. (Locally, `experiments/unpack_runs.py` does exactly that.)

- [ ] **Warn users that the reported results come from the withheld eval seeds, so their own videos and re-runs are of different instances.** Every reported number (`results.json`, the `runs`/`episodes`/`commits` tables, per-episode solved/failed) was measured on the 100 instances generated from the private eval seed, which dataset users don't have. Anything a user renders or re-evaluates is therefore on *other* instances, and its outcome need not match the reported one. Nothing says this today, and the viewer actively hides it: with `eval_seed: null` and no per-episode seeds in `results.json`, `_eval_seeds` falls back to the replicate seed (`experiments/results_viewer.py:896`), then shows the video for "episode i" (a different instance, with the same object count, so it isn't obvious) next to the recorded solved/failed of the real episode i. Suggested fixes: a sentence in the dataset card and README's reevaluation section, and in the viewer, label such renders as "not the scored instance" and show the rendered episode's own outcome instead of (or beside) the recorded one.

- [ ] **The archives aren't viewer-ready.** They ship `sandbox.bundle`, but the viewer needs `sandbox/.git` to show history (`has_history`). The dataset card's example clones the bundle into a separate `repo/`, which the viewer doesn't use. Restoring it with `git clone --no-checkout sandbox.bundle tmp && mv tmp/.git sandbox/.git` works.

- [ ] **25 source-access policies only run where the repo is mounted at `/robocode`.** Their sandboxes hard-code `/robocode/third-party/kindergarden/...` (14 `opus55-source`, 11 `opus5-source`; in 10 of them in the final `approach.py`), the container path used in the paper runs. Some use it only as a fallback, but it becomes the only path when the first lookup fails: e.g. `opus55-source/scooppour3d/seed222` (`kin.py`) first tries `os.path.dirname(kinder.envs.dynamic3d.__file__)`, but `kinder.envs.dynamic3d` is a namespace package (`__file__` is `None`), the resulting `TypeError` is swallowed, and outside the container it raises `FileNotFoundError: tidybot.xml`. So re-evaluating or rendering these policies on a host checkout fails (the viewer's render too) unless `/robocode` exists, e.g. `sudo ln -s <repo> /robocode`. Worth a note in the dataset card / README reevaluation section.

- [ ] **Dependency clash for reading the parquet tables.** They need `pyarrow`, which isn't a project dependency. The project pins NumPy 1.26, while current `pyarrow` (≥ 18) requires NumPy 2 and fails to import in the project venv. `pyarrow<18` works. The card's `load_dataset` example relies on `pyarrow` too (untested in the project venv).
