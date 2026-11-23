# Strategy traces of the dataset policies

A **trace** is the record of one episode of a policy: the state and the action of
each env step, and every contact with its exact on and off times. It has what is
necessary to see the strategy of the policy: where the bodies go, and what touches
what, in which order. This folder records traces of the policies in the dataset
`prpl-group/agentic-gentamp-runs` and draws an episode again from its trace,
without the policy.

All files here are new. They use `experiments/render_run.py` (`open_rollout`) and
`robocode.utils.episode`, and they change no file of the repository.

| File | Purpose |
| --- | --- |
| `record.py` | CLI: roll out a run at a commit on chosen instances, write the trace. **Runs policy code.** |
| `recorder.py` | The base class of a recorder, and the MuJoCo recorder: the contacts at each 0.5 ms substep (wraps `MjSim.step` and `MujocoEnv.step`) |
| `recorder2d.py` | pymunk: the contacts (arbiters) at each 10 ms substep, with the grasp as a contact |
| `recorder_kin2d.py` | Kinematic 2D: contacts from the geometry of each env step |
| `recorder_bullet.py`, `scenes_pddl.py` | Kinematic PyBullet, with scene adapters for the PR2 and the rovers |
| `backend.py` | Which recorder and which replay a scenario uses, and how to add a simulator |
| `entities.py` | The names of the bodies in a contact (`robot.gripper.left_pad`, `env.cube_0`) |
| `commits.py` | Which commits of a run change the policy, so an equal policy is recorded once |
| `trace.py` | Read a trace: the contacts as pairs of bodies, with times in seconds |
| `replay.py`, `replay_bullet.py`, `render2d.py` | Draw from a trace, no policy: episode video, contact video, ghost overlay of several episodes, labelled diagram |
| `motion.py` | The positions of the bodies over the episode (`motion.npz`) |
| `catalog.py` | The environment of a scenario from its Hydra config, its tags and its picture |

## The files of a trace

```
outputs/strategy_traces/<method>/<env>/seed<N>/commit<k>/inst<seed>[_count<c>]/
    trace.npz    contacts (keys, intervals, steps), qpos, ee, time, actions
    meta.json    run, commit, instance, outcome, the tables of the bodies
    motion.npz   positions of the robot base, the tool and each object (motion.py)
    video.mp4
outputs/strategy_traces/<method>/<env>/seed<N>/commits.json    the policy key of each commit
outputs/strategy_traces/_scenarios/<env>/                       pictures, diagrams, ghost videos
```

A contact key is (shape A, surface of A, shape B, surface of B). `intervals` has
the exact on and off times in substep ticks, so an impact shorter than one env step
is not lost. `steps` has one row for each env step in which a key is on: the share
of the step, the forces, and the contact point on each side. `qpos` is the state of
each env step, so a replay needs no policy.

Each run is recorded on the same instance seeds, so the traces of two runs are
comparable. They are not the scored episodes (the dataset withholds the eval seed).
Two records of one policy on one instance give the same trace.

## Use

```bash
# the final policy of each run of a scenario on instance 0, with video
uv run python -m experiments.strategy_traces.record '*/tossing3d/*' --instances 0
# other options: --commit k, --count c (set the object count), --force, --no-video

# each commit that changes the policy, and a sample of the others as a check
uv run python -m experiments.strategy_traces.record '*/tossing3d/*' --commits sample \
    --instances 0 --no-video --jobs 16

# videos from traces (no policy code)
uv run python -m experiments.strategy_traces.replay ghost <trace dir> <trace dir> ... -o out.mp4
uv run python -m experiments.strategy_traces.replay contact <trace dir> -o out.mp4

# positions of the bodies, pictures of the scenarios
uv run python -m experiments.strategy_traces.motion <trace dir> ...
uv run python -m experiments.strategy_traces.catalog
```

## Setup

- The parquet tables of the dataset need `uv pip install 'pyarrow<18'` (the project
  has NumPy 1.26). `uv sync` removes `pyarrow` again.
- The MimicLabs assets of KinDER (`scene_bg: mimiclabs-lab2`):
  `python third-party/kindergarden/scripts/download_mimiclabs_assets.py`.
- 25 policies with source access use the path `/robocode/third-party/kindergarden`:
  `sudo ln -s "$PWD" /robocode`.
- Recording runs the model-written policy code of the dataset in the process.
