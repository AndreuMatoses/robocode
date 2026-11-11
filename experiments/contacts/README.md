# Contact-strategy analysis (branch `feat/contact-strategy-analysis`)

Handoff note: current state, setup on a new machine, and next steps.

## Goal

Analyse the **contact strategies** of the policies in the dataset
`prpl-group/agentic-gentamp-runs` (the runs of the AgenticGenTAMP paper):

1. how diverse the strategies are across runs of one scenario (methods × seeds);
2. how a strategy evolves over an agent's run (across its git commits).

Scope so far:

- the 10 MuJoCo scenarios, all with the TidyBot (holonomic base, Kinova Gen3,
  Robotiq 2F-85): balancebeam3d, constrainedcupboard3d, dynamicshelf3d, dynamo3d,
  rearrange3d, scooppour3d, sortclutteredblocks3d, sweepintodrawer3d, sweepsimple3d,
  tossing3d;
- the 4 dynamic 2D scenarios (pymunk): dynobstruction2d, dynpushpullhook2d,
  dynpusht2d, dynscooppour2d (see "Simulator backends").

The kinematic scenarios (2D geometry, PyBullet, PR2, rovers) are listed in the
viewer but have no contact recording: they have no contacts, only collision checks
and attachments, and need a definition of "contact" first.

## What the branch adds over `main`

| Path | What |
|---|---|
| `experiments/unpack_runs.py` | Unpack dataset archives, restore `sandbox/.git`, make re-scored runs visible to the results viewer |
| `experiments/render_run.py` | Render a dataset run at any commit (MP4, `--scale`); `open_rollout()` builds env + policy and is shared with the recorder |
| `experiments/results_viewer.py` | Only change: fit-to-window toggle (`f`) in the video player |
| `experiments/contacts/entities.py` | Names for what touches what: `robot.gripper.left_pad`, `env.cube_0`, `env.table_1.top`, `world.floor`; class names without index (`env.cube`); surfaces (box faces `±x/±y/±z`, cylinder side/caps) and contact points in normalised local coordinates |
| `experiments/contacts/recorder.py` | Logs MuJoCo contacts at every 0.5 ms substep by wrapping `MjSim.step` / `MujocoEnv.step` |
| `experiments/contacts/backend.py` | The simulator families: which recorder and which replay a scenario or a trace uses (`mujoco`, `pymunk`), and how to add one |
| `experiments/contacts/recorder2d.py` | Logs pymunk contacts (arbiters) at every 10 ms substep of the dynamic 2D scenarios, with the grasp as a contact; subclass of the shared `recorder.BaseRecorder` |
| `experiments/contacts/render2d.py` | Draws a pymunk trace from the trace alone (no environment): episode video, contact video, ghost video, labelled diagram |
| `experiments/contacts/record.py` | CLI: roll out a run/commit on fixed instances, write trace + video + diagram |
| `experiments/contacts/replay.py` | Renders from recorded `qpos`, no policy: episode video, labelled diagram, contact-highlight video, ghost overlay |
| `experiments/contacts/trace.py` | Loads a trace and summarises it per entity pair (used by the viewer) |
| `experiments/contacts/commits.py` | Which commits of a run change the policy (hash of `approach.py`, the sandbox modules it imports and the data files it names); cached as `<run>/commits.json` |
| `experiments/contacts/histories.py` | Shared strategy vocabulary of a scenario (one identifier per distinct solved contact-mode sequence), run histories on a common axis, and the distance matrix between the versions of two runs |
| `experiments/contacts/strategy.py` | Contact-mode sequence of a trace (labels, filters, debounce) and two distances: Jaccard on the pair set, edit distance on the mode sequence. `Definition` = what counts as one strategy (which contacts, which names, which order), with presets |
| `experiments/contacts/diversity.py` | Final policies of a scenario: number of strategies, effective number (2^entropy), mean pairwise distance, within/between-method distance with a permutation test, 2-D map (classical MDS), episode length per method and per strategy |
| `experiments/contacts/metrics.py` | Six distances between two strategies (mode edit, elastic mode, pair set, contact duty, timeline overlap, time-warped), their rank correlations, the number of strategies against the threshold, and what is behind each number for one pair of runs |
| `experiments/contacts/repeat.py` | Repeatability of a strategy definition: the final policy of each run on several start scenes; within-policy and between-run distances, the thresholds they give, and the probability that two runs are farther apart than one policy is from itself |
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

## Simulator backends

The trace files have one layout for every simulator, so the analysis (`strategy`,
`histories`, `diversity`, `metrics`, `repeat`) and the viewer pages do not depend on
the simulator. A *backend* is the part that does: how to read the contacts and how
to draw a recorded episode again. `backend.py` names them and says how to add one;
`meta.json` has the key `"backend"` (a trace without it is a MuJoCo trace).

| backend | scenarios | recorder | replay |
|---|---|---|---|
| `mujoco` | the 10 dynamic3d scenarios | `recorder.ContactRecorder` | `replay.py`: sets `qpos` on a new environment and renders |
| `pymunk` | the 4 dynamic2d scenarios | `recorder2d.PymunkRecorder` | `render2d.py`: draws from the trace alone, no environment |

Both recorders are subclasses of `recorder.BaseRecorder`, which has the part that
does not depend on the simulator (intervals, per-step rows, the arrays). A trace of
the MuJoCo recorder is bit-identical before and after this split.

### pymunk (dynamic 2D)

- One `env.step` is 100 `Space.step` substeps of 10 ms: 1 s of simulated time. The
  recorder reads the arbiters (colliding shape pairs with points, normal, impulse)
  after every substep. Two records of one policy are identical (6 of 6 checked).
- The robot parts, the walls and the tables are kinematic bodies. When a robot part
  runs into a wall or a table, pymunk calls a handler of the environment, which
  moves the robot back to its last pose. No impulse is applied and the pair is in
  no arbiter list, so the recorder wraps those handlers and records the pair with a
  force of zero ("the robot is stopped by the wall"). In a sample of 7
  final policies of each scenario this is rare in dynobstruction2d (3 of 713 env
  steps), absent in dynpushpullhook2d, and 5 % of the env steps in dynscooppour2d
  (the arm on the middle wall).
- A grasp replaces the body of the object by a kinematic body that moves with the
  gripper. The recorder then adds the pairs finger - held object itself, with a
  force of zero (the grasp force is not simulated), so a grasp is one contact from
  the first touch to the release.
- Entities: `robot.base`, `robot.arm`, `robot.gripper.palm`,
  `robot.gripper.left_finger` / `right_finger` (`robot.dot` in dynpusht2d);
  `env.<name>` for objects (dynamic bodies) and fixtures (table, target surface);
  `world.<name>` for the walls. A "geom" is one pymunk shape; `qpos` holds the pose
  `x, y, theta` of every entity at every env step.
- To read the arbiters in Python is much slower than the simulation. A scene with
  12 or more movable bodies (dynscooppour2d: 30 objects, 35 times slower) is read
  at every 5th substep (50 ms; `substep_stride` in `meta.json`).
- The pictures show the collision shapes in the colours of the state objects, not
  the picture of `env.render()`. Episode video, contact video, ghost video and the
  labelled diagram all work from the viewer as for MuJoCo.

### Tuning for pymunk: not the MuJoCo values

The same two steps as for MuJoCo (gap histogram, repeatability test) were run on
the 2D data: the final policy of the 140 runs on instances 0 to 3 (560 episodes,
516 solved) and every policy version on instance 0 (3 130 records: 2 288 solve,
87 crash in the episode, 6 versions do not load; logs in
`outputs/contact_record_logs/dyn2d_*.log`).

- *Merge gap.* The robot moves at the start of an env step and then holds, so one
  push is a train of contacts of 0.1-0.3 s with gaps of 0.6-1.5 s (9 400 of the
  16 300 gaps of the three small scenarios, instances 1 to 3), only 330 gaps are
  between 1.5 s and 10 s, and a longer gap is a new contact. The gap is 3 s (3 env steps): it is the most
  repeatable value (mean AUC 0.693; 0.66-0.69 for 0, 0.3, 0.5, 1, 2, 5 and 10 s).
  `strategy.TIME_SCALE` holds the factor of each time for each backend (pymunk: gap
  × 10, the two other times × 1; they have no effect on the result there). The
  definition bar still shows the MuJoCo values; the page text gives the real ones.
- *Definition.* A policy changes more between two start scenes than in the MuJoCo
  scenarios, and the runs differ less from each other: the mean AUC is 0.69 at
  best (0.85 for MuJoCo). The pairs without a robot part (object on object, object
  on a wall) and the object names depend on the start scene. Within 0.25 are, of
  the pairs of one policy: 43 % with all pairs and names as given (the MuJoCo
  preset), 65 % with robot pairs only, 82 % with robot pairs and kinds only.
- *Preset "tuned 2D".* Robot pairs only, kind only, full mode sequence, elastic
  distance, threshold 0.35 (89 % of the pairs of one policy; the 90 % value is
  0.38 / 0.48 / 0.30 / 0.21 in dynobstruction2d / dynpushpullhook2d / dynpusht2d /
  dynscooppour2d, 0.37 pooled). On instance 0 it gives 5 / 2 / 3 / 2 strategies
  for 35 / 33 / 34 / 30 runs. A threshold of 0.25 gives 6 / 2 / 5 / 6 strategies
  and 82 %.
- *In the viewer.* A page of one scenario offers the tuned preset of its simulator
  with one button when it shows another definition (`strategy.TUNED_PRESET`), in
  both directions: "tuned 2D" on a 2D scenario, "tuned" on a MuJoCo scenario.
- *What it means.* In dynpushpullhook2d the AUC is 0.54: the runs are not more
  different from each other than one policy is from itself (all take the hook and
  pull). The strategies of the 2D scenarios are thus less reliable than those of
  the MuJoCo scenarios, and a diversity number there needs this caveat.

### Speed

The distances between many mode sequences are computed for all pairs at one time
(`strategy.key_distance_matrix`, the same values as the pair functions, 10 to 20
times faster), and the interval code has no Python loop over long contacts. The
first load of the history page of dynscooppour2d (534 traces with about 500 contact
keys each) takes about 15 s; it took more than 2 minutes before.

### Not done

The kinematic scenarios (kinematic2d, kinematic3d, PR2, rovers) have no contacts: a
motion into collision is rejected and a grasp is an attachment. A recorder for them
must first define a contact. The definition nearest to the MuJoCo one ("two bodies
touch at this instant") has three sources, each with a counterpart that is already
in the traces:

| source in a kinematic scenario | counterpart | what it gives |
|---|---|---|
| the two shapes are nearer than a small tolerance (PyBullet: closest points; 2D: distance of the geoms) | a MuJoCo contact | object on a table, object on object, the gripper at an object |
| the robot holds the object (attachment, suction) | the pad contacts of a MuJoCo grasp; the "finger - held object" pair of the pymunk recorder | the grasp, also when the gripper does not touch the object in the geometry |
| a motion is rejected because the robot or the held object would collide | a MuJoCo contact that stops the robot; the "stopped by the wall" pair of the pymunk recorder | the bump that never appears as an overlap |

What stays different: there is no force (so no impact rule and no impulse), the time
resolution is one env step, and the tolerance is a new parameter that must be tuned
like the merge gap (too small: a resting object does not touch its table; too
large: a near miss counts).

## State

- Recorded: all 350 runs of the 10 MuJoCo scenarios (7 methods × 5 seeds), final
  commit, instance 0, with video.
- Recorded, commit histories: all 10 scenarios, instance 0, every commit that changes
  the policy plus one unchanged commit per streak (`--commits sample`), no video:
  7 551 traces for 12 151 commits (5 957 policy versions). Logs:
  `outputs/contact_record_logs/tossing3d_commits.log` and `all_commits.log`.
- 38 policy versions have no trace (shown as "no trace"): 37 do not load (a `None`
  path, an import outside the sandbox allowlist, a syntax error, a module that the
  agent's environment did not have either, e.g. `threadpoolctl`), and
  `opus55-source/sweepsimple3d/seed42@1` hit the 30-minute limit of the batch (a
  retry without a limit was stopped after 10 hours). The
  other commit at the limit, `opus5-source/rearrange3d/seed24@1`, was recorded
  alone without a limit: 50 minutes of policy time for 1 000 steps, unsolved.
- Determinism check: 1 625 of the 1 632 re-recorded unchanged commits gave exactly
  the same trace. The 7 others have an identical policy (only logs and probe scripts
  differ); 4 of them still have the same mode sequence, 3 differ by an edit distance
  of 0.12–0.17, and none flips solved/unsolved. One of them
  (`opus55/sortclutteredblocks3d/seed444@20`, 296 steps in the batch run) gave 321
  steps in three later recordings on the idle machine, the same as `@21`. So a
  replay is not always bit-exact; the cause is not found (the batch ran 24 jobs on
  32 cores at a load of about 65).
- Entity names checked on every scenario: one scene signature per scenario, no
  unnamed entity.
- Evolution page (`#/e/<run>`, the "history" link of the run matrix): solve rate,
  strategy change and distance to the final strategy per commit; one strip of
  contact modes per policy version.
- Viewer works: index with thumbnails and tag filters; scenario page (run matrix,
  overlaid timelines with per-seed shades and force lines, labelled diagram, ghost
  video with a save button); run page (video and timeline with a shared playhead,
  diagram lighting up entities in contact, per-surface contact-point plots).
- Not exercised: the viewer's "Record missing" button and the run page's "contact
  highlight" toggle (the commands they launch work from the terminal).
- Histories page (`#/h/<scenario>`, linked from the scenario page): one lane per
  run, coloured by strategy identifier, x axis = fraction of the commits or time
  elapsed since the first commit; per-method table (first solve, settled, final
  strategies); strategy table; distance matrix for two picked runs.
- A strategy ends at the robot's last contact (`Options.trim_tail`): what the thrown
  cube touches afterwards depends on when the episode stops. With this, the solved
  tossing3d episodes have 34 distinct mode sequences; the most frequent is in 14 runs.
- Strategy classes on the histories page ("one strategy ="): similar mode sequences
  (complete-linkage clusters at edit distance 0.35; 19 classes in tossing3d), the
  exact sequence (34), or the same set of contact pairs (12).
- Strategy definition (`strategy.Definition`, the "One strategy =" bar of the
  histories and diversity pages; the page prints the definition in one sentence).
  Contacts: all pairs / robot pairs only. Names: as named / copies merged
  (`bin_red` = `bin_yellow`, `shelf2` = `shelf3`) / kind only (object, fixture,
  world; gripper, arm, base). Order: mode sequence / each mode once in order of
  first occurrence / each pair once in order of first occurrence / set of pairs.
  Same when: equal, or within a distance (0.2, 0.35, 0.5; complete linkage).
  Presets: detailed, object-agnostic, robot skeleton, contact set.
  Finger links: always count (default) / not while a pad holds
  (`Definition.fingers = "pad_first"`, `Options.pad_first`). The gripper has two
  kinds of parts: the *pads* (the two friction boxes at each fingertip) and the
  *finger* links (driver, coupler, spring link, follower). With "not while a pad
  holds", a finger link on a thing does not count while a pad of either side is on
  that same thing (same entity; the drawer face and its handle are two things). A
  finger link alone (a push or a sweep) still counts. Effect on the final policies
  (instance 0, detailed definition): in tossing3d all finger-object contact time is
  inside a pad contact on the same object and the mean edit distance drops from
  0.43 to 0.33 (11 → 9 strategies at 0.35); in rearrange3d 95 % is inside (0.63 →
  0.59, 17 → 15). In the other eight scenarios the mean distance changes by 0.01
  at most and the count at 0.35 does not change; in sweepintodrawer3d and
  constrainedcupboard3d only about a quarter of the finger contact time is inside
  a pad contact, so the finger contacts there are mostly pushes. The run page and
  the evolution page take no definition and still show the finger links.
- **Tuned definition (preset "tuned", the default of the pages).** All pairs, names
  as given (see *Names* below; the test below was run with copies merged), full
  mode sequence, finger links not while a pad holds, elastic mode
  distance (`warp`), same strategy within 0.25, merge gap 0.3 s. It is the result
  of a repeatability test (`repeat.py`), not of a look at the videos:
  - *Data.* The final policy of all 350 runs was recorded on instances 1, 2 and 3
    too (`record --commit -1 --instances 1 2 3 --no-video`; 1 050 episodes,
    `outputs/contact_record_logs/final_instances_1_2_3*.log`). A *within* pair is
    one policy on two start scenes; a *between* pair is two runs on two start
    scenes. Only scenes with the same number of objects are compared: the object
    count changes with the instance in constrainedcupboard3d (2/3), dynamicshelf3d
    (1/2), dynamo3d (1/3) and sweepsimple3d (1/5), and another count is another
    task (in dynamo3d with one object most policies have no contact at all).
  - *Criterion.* A definition is good when a policy is the same strategy as
    itself on another start scene. The threshold is the distance below which 90 %
    of the within pairs fall. The AUC is the probability that a between pair is
    farther apart than a within pair; it is used only to compare definitions of
    one family, because it rewards anything that identifies a policy (the contact
    duty has the highest AUC, 0.87–0.89, because it sees the speed of a policy).
  - *Distance.* The plain edit distance charges a full edit for a short extra mode
    that only adds one contact pair (pad on the floor, palm on the cube). Such
    modes come and go between two start scenes of one policy. The elastic mode
    distance (`strategy.warp_distance`: dynamic time warping over the modes, not
    over time; a matched pair of modes costs their Jaccard distance; mean cost
    along the best alignment) charges only the Jaccard difference. 90 % of the
    within pairs are within 0.23 with the elastic distance and within 0.47 with the
    edit distance (pooled over the ten scenarios). At 0.35 the edit distance
    splits one policy from itself in 23 % of the within pairs. The first default
    of this branch (edit distance, names as given, finger links always, merge gap
    0.05 s, threshold 0.35) did so in 34 %; the tuned definition does so in 8 %.
  - *Threshold.* With the elastic distance the 90 % value is 0.13–0.30 in nine
    scenarios (dynamo3d 0.42), so one threshold serves all: at 0.25, 92 % of the
    within pairs are one strategy (per scenario 0.87–1.00; dynamo3d 0.80). On
    instance 0 it gives 3 / 4 / 5 / 6 / 9 / 13 / 4 / 17 / 6 / 7 strategies
    (balancebeam, constrainedcupboard, dynamicshelf, dynamo, rearrange, scooppour,
    sortclutteredblocks, sweepintodrawer, sweepsimple, tossing). The many
    strategies of sweepintodrawer3d and scooppour3d are real: the AUC there is
    0.95–0.97, so the runs differ far more from each other than one policy from
    itself. In dynamo3d the AUC is 0.58: the runs are not more different from each
    other than one policy is from itself, so the strategies there are not reliable.
  - *Definition.* Copies merged is better than as named (sortclutteredblocks3d:
    AUC 0.79 against 0.61, the block colours change with the instance); kind only
    is too coarse (rearrange3d 0.67); the full sequence is better than the
    first-occurrence orders. The finger rule is neutral in this test (mean AUC
    0.852 with the rule, 0.858 without; 92 % of the within pairs at 0.25 in both
    cases), because the elastic distance already makes an extra finger pair cheap.
    It is on in the preset as a choice of meaning, not as a result of the test.
  - *Names.* The preset keeps the names as given (a choice of the user): on one
    instance all runs see the same objects, so the names say which object a run
    handles first. Against copies merged, nine scenarios do not change (rearrange3d
    has 10 strategies and not 9). sortclutteredblocks3d changes: its block names
    change with the instance, so only 22 % of its within pairs are within 0.25
    (96 % with copies merged), the AUC is 0.63 (0.80), and instance 0 has 12
    strategies (4). Pooled, 83 % of the within pairs are within 0.25 (92 %). Use
    "copies merged" for sortclutteredblocks3d when the order of the block colours
    is not part of the strategy, and for a comparison across instances.
  - *Robot on the floor.* Without the robot-world pairs (pad on the floor) the AUC
    drops from 0.852 to 0.813 (tossing3d 0.80 to 0.64). These contacts are a
    repeatable property of a policy, not noise, so they stay in. What made them a
    problem was the full edit that the edit distance charged for them.
  - *Debounce.* Of the raw gaps between two contacts of one pair, 86 % are below
    5 ms and 96 % below 30 ms (contact chatter); few are between 0.2 s and 1 s
    (about 100 in each of the bins 0.2–0.3, 0.3–0.5 and 0.5–1 s, against 523 in
    0.1–0.2 s and 771 above 2 s); a long gap is a new contact. The merge gap is now 0.3 s (`Options.gap`, was 0.05): the
    mean AUC goes from 0.845 to 0.852 (edit distance: 0.837 to 0.848, and its mean
    within distance from 0.24 to 0.19); 0.1 s to 1 s all give 0.849–0.852. The
    other times stay: a longer shortest contact changes nothing or lowers the AUC
    (0.834 at 0.3 s), and so does a longer shortest mode (0.852 at 0.05 s, 0.845 at
    0.1 s, 0.833 at 0.2 s); with the old gap a shortest mode of 0.2 s also removed
    the base impacts of dynamo3d, which last 10–50 ms. The impulse of a contact
    (mean normal force × time, `Options.keep_impulse`) separates those impacts
    well (above 3 N s; most other short contacts are below) but gives the same
    result as the present force rule, so it is off by default. An impact that is
    kept by one of the two rules now counts as a contact of at least the shortest
    mode time, so that its mode is not absorbed.
  - *Base contacts.* Every robot part counts in every scenario (base, arm, palm,
    finger links, pads). The base touches something in a few runs only: tossing3d
    2 of 30 runs (the barrier, 1 s in total), rearrange3d 5 of 32 (the base stays
    against a fixture, 367 s in total), dynamo3d all runs (the task).
  The definition bar has the distance (edit, elastic, timeline overlap) and, in a
  second row, the debounce times. The explorer page `#/m/<scenario>` has a
  repeatability panel: the within and between distances of the selected distance,
  the 50 / 80 / 90 % values (buttons set the slider) and the AUC.
- Distance explorer: `#/m/<scenario>` (link on the diversity page). The distance of
  the strategy classes is **not** a plain Jaccard distance: it is an edit distance
  between the two contact-mode sequences, where to replace a mode costs the Jaccard
  distance between the two modes, divided by the length of the longer sequence
  (only the definition "set of pairs" reduces to Jaccard). The page puts it next to
  five other distances (`metrics.py`): the elastic mode distance (see the tuned
  definition above), pair set, contact duty (weighted Jaccard of
  the fraction of time each pair is in contact), timeline overlap (temporal IoU) and
  a time-warped mode distance (DTW). A slider sets a distance: the page shows the
  three pairs of runs nearest to it, each with the contact pairs over time, the
  aligned mode sequences with the cost of each column, and the arithmetic of every
  distance; the histogram of all pair distances; the number of strategies against
  the threshold for every metric; a scatter of two metrics (click a dot for the
  pair) and their Spearman correlations. A ghost video of the two runs can be
  rendered from each pair.
  First numbers (final policies, instance 0, detailed definition, merge gap
  0.05 s as it was then):
  - The scales differ. Mean distance in tossing3d: edit 0.43, set 0.32, duty 0.27,
    overlap 0.28, dtw 0.12. One threshold does not fit all: 0.35 gives 11 / 5 / 4 /
    7 / 2 strategies there.
  - duty and overlap order the pairs nearly the same (Spearman 0.81–0.99 over the
    ten scenarios); one of them is enough.
  - edit and set agree least in sortclutteredblocks3d (0.40) and most in
    scooppour3d (0.85): with many objects the order carries information that the
    set does not have.
  - The edit distance has the largest mean in 6 of 10 scenarios (a tie with set
    in one; timeline overlap is larger in dynamo3d, sweepintodrawer3d and
    sweepsimple3d). It can be 0.33 for two runs with the same pair set and a
    time-warped distance of 0.00: a short extra mode costs a full edit.
  - Not built: metrics on geometry (contact location on the object, end-effector
    path), sliders for the debounce thresholds and for the edit costs.
- Diversity pages: `#/d` (all scenarios × methods, one metric at a time, and how
  often the source-access method is higher / equal / lower than its counterpart)
  and `#/d/<scenario>` (tiles, map of the final strategies, run × run distance
  matrix, method × method mean distance, per-method table, episode length per
  method or per strategy, strategy table). The population is the final policy of
  each run that solves instance 0.
- Optimality: on the 10 MuJoCo scenarios the reward is −1 per step (checked on the
  recorded traces), so total reward = −episode length; one step is 0.1 s. The pages
  show the episode length of the final policy, of the first policy version that
  solves the instance, and the compute time of the policy during the episode.
- First numbers (detailed definition, instance 0): the mean distance between two
  runs of one method is below the distance between runs of different methods in
  all 10 scenarios (permutation p < 0.05 in 7), so the method explains part of the
  strategy. Under the detailed definition a method's 5 runs are mostly 4–5
  different strategies in the multi-object scenarios; the robot-skeleton and
  kind-only definitions bring that down (dynamo3d: 4 strategies for 34 runs).
- Not built: the finer distances (contact location, force profile, DTW); diversity
  over all solved policy versions (what an agent explored), only the final ones.
- The diversity pages were tested through the DOM (content, no script error, no
  overflow, marks inside their plots) and looked at in screenshots of tossing3d and
  of the overview. For some hours the headless Chromium used for screenshots
  produced no frames on this machine; it worked again later and the cause is not
  known.
- Ghost video of the final policies (diversity page of a scenario): the runs in one
  scene, one translucent robot per run, coloured by strategy or by method. The runs
  are the shortest episode of each strategy, of each method, or all final policies
  that solve the instance. No run is the reference (`replay ghost --equal`). An
  object or a drawer is ghosted only while its pose differs between the runs. All
  runs start together and a finished run holds its last pose, so the video compares
  motion from the start of the episode, not matching phases. The camera is the
  scenario's; a robot can leave the frame (sweepintodrawer3d). Strategies after the
  12th most common are grey. 30 runs of tossing3d render in about 3 s.
  For the diversity inside a strategy, the same panel has two more choices of runs:
  "all runs of one strategy" (pick the strategy; one video) and "all runs of each
  strategy, one video for each (grid)" (one tile per strategy with 2 or more runs;
  a checkbox adds the strategies with one run; "Play all from the start" restarts
  the tiles together). There the colour is per run or per method. The strategies
  are those of the definition bar, so a change of the definition or the threshold
  changes the tiles. A ghost video that is newer than its traces and than
  `replay.py` is reused and not rendered again (`viewer.start_job`).
- Strategy classes saturate on the multi-object scenarios: similar / exact / pairs
  give 89 / 212 / 94 classes in sortclutteredblocks3d and 47 / 110 / 41 in
  sweepintodrawer3d, where the most frequent class holds 2 of 35 runs. A mode
  sequence there has tens of modes and encodes the order in which the objects are
  handled, so nearly every run is its own class and "distinct final strategies per
  method" is mostly 4–5 of 5. The tossing3d observation "source access gives more
  final variety" does not hold across scenarios. A coarser representation (collapse
  repeats per object, ignore the object identity) or a continuous measure (mean
  pairwise distance within / across methods) is needed before comparing methods.
- Known limits of the pilot (written before the tuned definition above, which
  replaces the first guesses of the merge gap and of the threshold): the debounce
  thresholds in `strategy.Options` are first
  guesses and some traces still flicker between two modes; a tossing3d sequence has
  only 5–8 modes, so one changed mode moves the edit distance by 0.2 or more; mode
  colours repeat after 12 modes; unsolved episodes get no strategy identifier;
  elapsed time is wall time between commits; the grouping threshold (0.35) is a
  first guess, checked by eye on tossing3d only.
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

# record the history of each run: every commit that changes the policy (+ a sample
# of unchanged ones as a determinism check), traces only, 16 at a time
uv run python -m experiments.contacts.record '*/tossing3d/*' --commits sample \
    --instances 0 --no-video --jobs 16

# strategy over the commits in the terminal; mode sequence of one trace
uv run python -m experiments.contacts.strategy evolution 'opus55/tossing3d/*'
uv run python -m experiments.contacts.strategy show <trace dir>

# viewer at http://127.0.0.1:8010
uv run python -m experiments.contacts.viewer

# renders from traces
uv run python -m experiments.contacts.replay ghost <trace dir> <trace dir> ... -o out.mp4

# more start scenes of the final policies, for the repeatability test (repeat.py)
uv run python -m experiments.contacts.record '*/tossing3d/*' --commit -1 --instances 1 2 3 --no-video --jobs 12
#   --all-bodies: ghost the objects too; --equal: no solid reference run; --alpha 0.3
uv run python -m experiments.contacts.replay contact <trace dir> -o out.mp4
```

Output layout (`outputs/contact_traces/`):

```
<method>/<env>/seed<N>/commit<k>/inst<seed>[_count<c>]/  trace.npz, meta.json, video.mp4
<method>/<env>/seed<N>/commits.json     policy key of every commit (commits.py)
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
2. Pick the strategy definition to report (see the definition bar; the choice
   changes the counts, much less the mean distances).
3. Find why a replay is sometimes not bit-exact (see the determinism check in
   "State"); record a few commits several times with and without load.
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
