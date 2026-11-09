# Contact-strategy analysis (branch `feat/contact-strategy-analysis`)

Handoff note: current state, setup on a new machine, and next steps.

## Goal

Analyse the **contact strategies** of the policies in the dataset
`prpl-group/agentic-gentamp-runs` (the runs of the AgenticGenTAMP paper):

1. how diverse the strategies are across runs of one scenario (methods × seeds);
2. how a strategy evolves over an agent's run (across its git commits).

Scope so far: the 10 MuJoCo scenarios, all with the TidyBot (holonomic base, Kinova
Gen3, Robotiq 2F-85): balancebeam3d, constrainedcupboard3d, dynamicshelf3d, dynamo3d,
rearrange3d, scooppour3d, sortclutteredblocks3d, sweepintodrawer3d, sweepsimple3d,
tossing3d. The 2D and PyBullet scenarios are listed in the viewer but have no contact
recording.

## What the branch adds over `main`

| Path | What |
|---|---|
| `experiments/unpack_runs.py` | Unpack dataset archives, restore `sandbox/.git`, make re-scored runs visible to the results viewer |
| `experiments/render_run.py` | Render a dataset run at any commit (MP4, `--scale`); `open_rollout()` builds env + policy and is shared with the recorder |
| `experiments/results_viewer.py` | Only change: fit-to-window toggle (`f`) in the video player |
| `experiments/contacts/entities.py` | Names for what touches what: `robot.gripper.left_pad`, `env.cube_0`, `env.table_1.top`, `world.floor`; class names without index (`env.cube`); surfaces (box faces `±x/±y/±z`, cylinder side/caps) and contact points in normalised local coordinates |
| `experiments/contacts/recorder.py` | Logs MuJoCo contacts at every 0.5 ms substep by wrapping `MjSim.step` / `MujocoEnv.step` |
| `experiments/contacts/record.py` | CLI: roll out a run/commit on fixed instances, write trace + video + diagram |
| `experiments/contacts/replay.py` | Renders from recorded `qpos`, no policy: episode video, labelled diagram, contact-highlight video, ghost overlay |
| `experiments/contacts/trace.py` | Loads a trace and summarises it per entity pair (used by the viewer) |
| `experiments/contacts/catalog.py` | Scenario tags (simulator, robot, dimension, dynamics) and thumbnails |
| `experiments/contacts/viewer.py`, `static/` | The contact viewer (stdlib HTTP server, vanilla JS) |

Design points worth knowing:

- **Record once, replay freely.** A trace stores `qpos` per env step, and
  `reset(seed, count)` is deterministic, so every image and video is re-rendered from
  the trace. Policy code only runs in `record.py`.
- Contacts are keyed by (geom A, surface A, geom B, surface B). Stored: exact on/off
  intervals in substep ticks, and per env step the active fraction, mean/peak normal
  force, tangential force, and the force-weighted contact point on each side.
- Every run is recorded on the **same instance seeds**, so traces are comparable and
  can be overlaid as ghosts. These are not the scored episodes: the dataset withholds
  the eval seed.
- A pair is "resting" (hidden by default) if it involves no robot part and is active
  for ≥ 98 % of the episode. Fixture-internal contacts (drawers against their cabinet)
  show very large forces and are also hidden by default.

## State

- Recorded: all 35 tossing3d runs (7 methods × 5 seeds), final commit, instance 0.
  30 solved; the five `sol` runs did not. Nothing else is recorded.
- Viewer works: index with thumbnails and tag filters; scenario page (run matrix,
  overlaid timelines with per-seed shades and force lines, labelled diagram, ghost
  video with a save button); run page (video and timeline with a shared playhead,
  diagram lighting up entities in contact, per-surface contact-point plots).
- Not exercised: the viewer's "Record missing" button and the run page's "contact
  highlight" toggle (the commands they launch work from the terminal).
- Not built: diversity metrics, evolution-across-commits view.
- Lint: only `black`, `isort` and a basic `ruff` pass were run; the repo's `mypy` and
  `pylint` (`./run_ci_checks.sh`, needs the `develop` extra) were not.
- `things_to_fix.md` (repo root, untracked on purpose) lists dataset/viewer issues to
  report upstream.

## Setup on a new machine

```bash
git clone <your fork> robocode && cd robocode
git checkout feat/contact-strategy-analysis
git submodule update --init --recursive
uv sync --dev
uv pip install 'pyarrow<18'      # parquet tables; the project pins NumPy 1.26
python third-party/kindergarden/scripts/download_mimiclabs_assets.py   # ~1 GB
```

- `uv sync` removes `pyarrow` again (it is not a project dependency); reinstall it
  after any sync. `uv run` keeps it.
- The MimicLabs assets are needed by constrainedcupboard3d (`scene_bg: mimiclabs-lab2`).
- 25 source-access policies hard-code `/robocode/third-party/kindergarden/...`:
  `sudo ln -s "$PWD" /robocode`.
- Rendering needs a working MuJoCo GL backend (EGL here); the repo picks it in
  `robocode.environments.mujoco_gl.configure_gl_backend`.

Dataset (public; revision used: `7dec3d08`; about 8 GB packed, 14 GB with the seven
agentic methods unpacked):

```bash
hf download prpl-group/agentic-gentamp-runs --repo-type dataset \
    --revision 7dec3d08 --local-dir viewer_runs/agentic-gentamp-runs
uv run python experiments/unpack_runs.py '*/tossing3d/*'    # or any glob; '*/*/*' for all
```

`record.py` unpacks a run's archive on first use, so the unpack step is only needed
for browsing with `experiments/results_viewer.py`. `llmgenplan` and `planner` have no
commits/programs and are not used here.

`viewer_runs/` and `outputs/` are git-ignored: the dataset and all traces are local
and must be re-created (or copied) on the new machine.

## Usage

```bash
# thumbnails for the index page (no policy code)
uv run python -m experiments.contacts.catalog

# record: final commit, instance 0, all runs of a scenario (one subprocess per run)
uv run python -m experiments.contacts.record '*/tossing3d/*' --instances 0
# other options: --commit k, --count c (pin object count), --force

# viewer at http://127.0.0.1:8010
uv run python -m experiments.contacts.viewer

# renders from traces
uv run python -m experiments.contacts.replay ghost <trace dir> <trace dir> ... -o out.mp4
uv run python -m experiments.contacts.replay contact <trace dir> -o out.mp4
```

Output layout (`outputs/contact_traces/`):

```
<method>/<env>/seed<N>/commit<k>/inst<seed>[_count<c>]/  trace.npz, meta.json, video.mp4
_scenarios/<env>/thumb.png
_scenarios/<env>/inst<seed>[_count<c>]/   <view>.png, <view>_ids.png, diagram.json
_scenarios/<env>/ghosts/<hash>.mp4
```

Recording executes the dataset's model-written policy code in-process. For an agent
in Claude Code this needs the allow rule
`Bash(uv run python -m experiments.contacts.record:*)`.

## Next steps

1. Record more of tossing3d (instances 1, 2; a pinned 2-cube instance) and check that
   the method differences seen on instance 0 hold.
2. Record the other nine MuJoCo scenarios; check entity naming on each.
3. Diversity metrics (see "Planned analysis" below; nothing is implemented yet).
4. Evolution view and metrics across commits (same section).
5. Later: 2D (pymunk) and PyBullet scenarios need their own contact readers.

## Planned analysis (ideas so far, none implemented)

These are proposals to refine, not decisions. The recorded data already contains
everything they need.

### Representing a trace

- **Contact-mode sequence.** At each instant the *mode* is the set of active contact
  pairs. A trace becomes the sequence of distinct modes, with durations dropped
  (optionally kept as weights). This is the symbolic "contact strategy".
- **Level of abstraction** (the same three levels the viewer's "rows by" offers):
  - class level (`robot.gripper.left_pad ↔ env.cube`): comparable across instances
    with different object counts, and invariant to which cube is handled first;
  - entity level (`… ↔ env.cube_0`): keeps object identity and order;
  - surface level (adds geom index and face): keeps *how* the object is touched.
- **Debouncing.** Raw intervals flicker (hundreds of sub-100 ms intervals per pair).
  Proposed: merge gaps shorter than ~50 ms, drop blips shorter than a threshold unless
  their force is high (impacts matter for tossing), possibly hysteresis on force
  rather than a single threshold. The thresholds need tuning per scenario; check
  their effect in the viewer before trusting any metric.
- **Which contacts count.** Default: robot-involving pairs plus object contacts that
  change during the episode; exclude resting contacts and fixture-internal ones (as
  the viewer does). Robot self-contacts (finger linkage) are probably noise.
- **Left/right symmetry.** Consider merging `left_pad`/`right_pad` (and the fingers)
  into one class, so mirrored grasps are not counted as different strategies.

### Comparing two traces

- **Edit distance** (Levenshtein) between contact-mode sequences: order-sensitive,
  duration-free. A weighted variant can charge by the Jaccard distance between the
  two modes instead of 0/1.
- **Jaccard distance over the set of pairs (or surfaces) ever used:** order-free;
  answers "does this run use contacts the other never does" (e.g. finger links on the
  cube, pads on the floor, base against the barrier).
- **DTW on the pair × time binary matrix** (or on force per pair): when timing and
  duration matter, tolerant to speed differences.
- **Contact location distributions per surface:** where on the pad, which cube face,
  using the stored normalised local points; compare with a histogram distance
  (e.g. earth mover's). Captures grasp variety that the symbolic sequence hides.
- **Force profile features per pair:** peak and mean normal force, impulse, time in
  contact, number of intervals; tangential-to-normal ratio as a slip/push indicator.
- **Kinematic complement** (from the stored `qpos` and `ee`): distance between
  end-effector or base paths, to separate "same contacts, different motion" from
  "different contacts". The ghost video is the qualitative version of this.

### Diversity of a scenario

- Mean pairwise distance, and number of clusters (hierarchical or k-medoids on the
  distance matrix), computed
  - within a method across its 5 seeds (does the same agent find the same strategy?),
  - across methods (do models differ; does source access change the strategy?).
- Entropy over cluster membership as a single diversity number.
- Compare always on the **same instances**; average over several instance seeds and
  pinned counts. Separate solved from unsolved episodes (a failed rollout is a
  different kind of "diverse").
- Relate diversity to performance: solve rate per cluster, steps to solve.

### Evolution over an agent's run

- Record the same fixed instances at several (or all) commits of a run.
- Per commit: distance to the previous commit and to the final policy; plot against
  commit index, `reeval_solve_rate` (in the `commits` table) and commit time.
- Look for: early convergence vs late strategy switches, whether solve-rate jumps
  coincide with contact-strategy changes or only with parameter tuning (same
  contacts, different timing/forces), and when each contact pair first appears.
- Viewer: lanes per commit in the timeline, a commit strip (reuse `_snapshots`,
  `_approach_diff`, `_why` from `experiments/results_viewer.py`), ghost video across
  commits.
- Cost: runs have up to ~150 commits; subsample (e.g. commits where `approach.py`
  changed, or every k-th) before recording everything.

### Open points on the contact representation

- **Mesh surfaces.** MuJoCo collides convex hulls and reports no mesh face id, so
  mesh geoms (the robot links) are a single "surface" with a normalised local point.
  A nearest-face (or nearest-vertex) lookup on the mesh could be added if finer
  robot-side location is needed. Pads and all task objects seen so far are
  primitives, so this mainly affects arm links.
- **Surface frame.** Currently the geom frame, origin at the shape centre, scaled by
  half-extents (meshes: their bounding box; planes: metres, unscaled). Curved
  surfaces could instead use angle/height (cylinder) or azimuth/elevation (sphere).
- **Contact point.** MuJoCo's contact position is the midpoint between the two
  surfaces; per env step we keep the force-weighted mean over substeps and over the
  contact points of a pair (a box on a plane has four), so multi-point contacts
  collapse to their centre of pressure.
- **Temporal resolution.** Intervals are exact to 0.5 ms; forces and points are per
  env step (0.1 s). Store per-substep forces only if impacts need it.
- **Viewer ideas not done:** collapsible hierarchy in the timeline rows, labels with
  leader lines instead of numbered badges, contact points/force arrows drawn in the
  contact video.
