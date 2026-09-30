# Online Continual Learning for Robots in Dynamic Environments: A Non-Axiomatic Reasoning Approach

Code for the MSc thesis of the same title. The thesis evaluates OpenNARS for
Applications (ONA) as an online continual learner in a symbolic sorting task
abstracted from a robot arm, compares it with contextual bandit methods, and
connects the same controllers to a simulated Franka Emika Panda arm through
ROS 2 and MoveIt 2.

## Demo video

[`demo/RQ3A_relational_ona.mp4`](demo/RQ3A_relational_ona.mp4) records ONA with
relational encoding controlling the simulated arm in Experiment 3a (about 1 h 55 min).
RViz shows the sphere, the destination bins, and the indicator light. The color
names above the bins and the status panel are for viewers only; the controller
does not observe them.

## Repository layout

| Path | Content |
|---|---|
| `src/adaptive_sorting/` | Task environment, controllers (ONA and bandit methods), experiment runners, analysis |
| `src/adaptive_sorting/configs/task_rules.yaml` | Color–bin mappings of all experiments |
| `tests/` | Unit tests |
| `third_party/OpenNARS-for-Applications/` | ONA v0.9.3 source with the modifications described below |
| `ros2/` | ROS 2 package `adaptive_sorting_bringup` and its build and launch scripts |
| `demo/` | Demo video |

## Requirements

- Linux x86_64 with GCC and git. The experiments ran on Ubuntu 24.04 (WSL2) with GCC 13.3.
- Conda with Python 3.12, which also matches the Python of ROS 2 Jazzy. The ROS 2
  scripts run the experiment code in a Conda environment named `thesis`.
- A git clone of this repository: the ONA build records the source identity with `git`.
- Memory: each ONA process uses up to about 3.5 GB with the capacities used here, and
  up to three run in parallel by default. Use `--workers 1` or `2` if memory is short.
- Disk: the full set of thesis runs takes about 14 GB, mainly trial logs, the
  per-trial memory snapshots of ONA (relational) in Experiments 3a and 3b, and the
  per-seed checkpoints under `seeds/`. The checkpoints repeat the trial logs and
  can be deleted once a run has completed, which saves about 40%.
- For the robot demo only: ROS 2 Jazzy with MoveIt 2.

## Setup

```bash
git clone https://github.com/Noctis-Xu/ona-online-continual-learning.git
cd ona-online-continual-learning
conda create -n thesis python=3.12
conda activate thesis
pip install -e . pytest
```

Build ONA. The experiments use ONA compiled with compound-term capacity 256 and
implication-table capacity 240; ONA with temporal implication projection
(ONA-TIP) is built from the same source:

```bash
python -m adaptive_sorting.experiments.ona_profile
python -m adaptive_sorting.experiments.ona_profile --variant ona-tip
```

This produces `third_party/OpenNARS-for-Applications/NAR-c256-t240` (ONA, the
default engine of all experiments) and `NAR-tip-c256-t240` (ONA-TIP). Each binary gets a `.profile.json` with its compiler, build
command, capacities, source hash, and SHA-256; every experiment run copies this
record into its metadata. Add `--force` to rebuild an existing binary.

Then run the tests and a short smoke test of every experiment and controller
with two shortened runs each:

```bash
pytest
python -m adaptive_sorting.experiments.run_rq_comparison all --seeds 2 --trials 300/300
```

All commands run from the repository root in the activated `thesis` environment.

Changes to ONA v0.9.3 (upstream commit in
[`UPSTREAM.md`](third_party/OpenNARS-for-Applications/UPSTREAM.md)):

- a command-line option that sets the pseudorandom seed of each run;
- the compile-time option `TEMPORAL_IMPLICATION_PROJECTION`, off by default and
  on only in ONA-TIP, which projects the confidence of temporal implications with
  a fixed half-life of 10,000 steps (thesis, Appendix A).

## Run the experiments

```bash
python -m adaptive_sorting.experiments.run_rq_comparison EXPERIMENT... [AGENT...] [options]
```

| Thesis | Experiment ID | Trials per run |
|---|---|---|
| Experiment 1: initial learning | `rq1` | 1400 |
| Experiment 2a: single change of 2, 3, or 4 colors | `rq2a` | 200 + 1200, for each change |
| Experiment 2b: repeated changes A→B→C→B→A→B→C→B→A | `rq2b` | 200 + 8 × 1200 |
| Experiment 3a: relevant light L1 | `rq3a` | 200 + 1200 |
| Experiment 3b: L1 and irrelevant light L2 | `rq3b` | 200 + 1200 |
| Experiment 4a: new colors, numbered bins | `rq4a` | 200 + 1200 |
| Experiment 4b: new colors, color-labeled bins | `rq4b` | 200 + 1200 |

`all` selects all seven experiments. Without agent IDs, all five controllers run
one after another:

| Agent ID | Thesis label |
|---|---|
| `flat_ona` | ONA (flat) |
| `relational_ona` | ONA (relational) |
| `epsilon_greedy` | Epsilon-greedy |
| `ucb1` | Contextual UCB1 |
| `sw_ucb` | Contextual SW-UCB |

Options:

- `--seeds N` runs seeds 1 to N (default 10). The thesis uses 200.
- `--workers N` sets the number of parallel seeds (default 3, at most 3 for ONA).
- `--trials INITIAL/SUBSEQUENT` shortens or lengthens the phases; Experiment 1 uses
  only `INITIAL`. Each experiment enforces minimum lengths and reports them on invalid input.
- `--ona-binary PATH` selects the ONA engine for both encodings.

Each run writes to `experiment_logs/<experiment>/<timestamp>/`, with one
subdirectory per controller. A controller directory holds `metadata.json`
(parameters, seeds, task configuration, engine identity), `trials.csv` with one
row per trial, per-seed results under `seeds/`, and figures. The run directory
holds the cross-controller metrics (`core_metrics.csv`, `per_seed_metrics.csv`,
`paired_differences.csv`) and learning curves. The metrics and confidence
intervals follow the definitions in Chapter 3 of the thesis.

## Reproducing the thesis results

All results use seeds 1–200. Each 200-seed command takes hours; on the machine
used for the thesis, the main comparison took about eight hours with three workers.

| Thesis result | Command |
|---|---|
| Experiments 1–4b (Chapter 4) | `run_rq_comparison all --seeds 200` |
| ONA-TIP in Experiment 2b (Chapter 4) and in the other experiments (Appendix B) | `run_rq_comparison all flat_ona relational_ona --seeds 200 --ona-binary third_party/OpenNARS-for-Applications/NAR-tip-c256-t240` |
| Experiments 3a and 3b with a mixed phase of 4800 trials (Appendix B) | `run_rq_comparison rq3a rq3b relational_ona --seeds 200 --trials 200/4800` |
| Implication traces of ONA and ONA-TIP in Experiment 2b (Chapter 4) | `trace_rq2b_evidence --agent relational_ona --seed 1 --snapshot-interval 1 --ona-binary <engine>`, once with each engine |
| Suppression threshold intervention (Appendix B) | See [below](#suppression-threshold-intervention) |
| Metrics, figures, and stuck mappings (Chapter 4 and Appendix B) | See [below](#tables-and-figures) |

Prefix the commands with `python -m adaptive_sorting.experiments.`.

- In Experiments 3a and 3b, ONA (relational) also writes its retained implications
  after every trial to `memory/`, which the decision-source and implication
  figures use. Reading memory does not run inference, so the trajectory is the
  same as without it.
- The implication trace replays run 1 of Experiment 2b with the full phase lengths,
  takes the same actions as that run, and writes the implications behind each
  decision to `evidence_trace.jsonl` in `experiment_logs/rq2b/<timestamp>/`.
- Appendix A reports ONA's own unit and system tests for ONA-TIP. Run a binary
  without arguments in `third_party/OpenNARS-for-Applications/` to execute them:
  `./NAR-c256-t240` passes all tests, and `./NAR-tip-c256-t240` passes the unit
  tests and stops at `Sequence_Test`, the one system test it fails (Appendix A).

### Suppression threshold intervention

This intervention raises the suppression threshold of motor babbling, the
expectation below which ONA replaces a decision by a random action, from 0.55 to 0.65.
Set the value in `third_party/OpenNARS-for-Applications/src/Config.h`:

```c
#define MOTOR_BABBLING_SUPPRESSION_THRESHOLD 0.65
```

then build it as a separate engine, restore `Config.h`, and run seeds 1–20:

```bash
python -m adaptive_sorting.experiments.ona_profile --profile ona-c256-t240-babbling065 \
  --output third_party/OpenNARS-for-Applications/NAR-c256-t240-babbling065
git checkout third_party/OpenNARS-for-Applications/src/Config.h
python -m adaptive_sorting.experiments.run_rq_comparison rq3a rq3b relational_ona --seeds 20 \
  --ona-binary third_party/OpenNARS-for-Applications/NAR-c256-t240-babbling065
```

The thesis compares these runs with seeds 1–20 of the main runs. The
`decision_source` column of the modified runs labels motor babbling with the default
threshold, which is set separately in `src/adaptive_sorting/agents/ona_agent.py`;
accuracy and actions are unaffected.

### Tables and figures

The metrics and intervals of Chapter 4 and Appendix B come from one evaluation
report over the runs above. List the `trials.csv` of each controller per experiment
in a manifest, with the labels of the agent table and `ONA-TIP (flat)` or
`ONA-TIP (relational)` for the ONA-TIP runs:

```json
{
  "rq2b": {
    "sources": {
      "ONA (relational)": "experiment_logs/rq2b/RUN_A/relational_ona/trials.csv",
      "ONA-TIP (relational)": "experiment_logs/rq2b/RUN_B/relational_ona/trials.csv"
    }
  }
}
```

```bash
python -m adaptive_sorting.analysis.generate_report --manifest sources.json \
  --output-dir experiment_logs/evaluation/NAME
```

Relative paths resolve from the manifest's directory. The report holds the core
metrics with 95% intervals (including trials to criterion for each switch of
Experiment 2b), learning curves, the decision sources and retained implications
of ONA (relational) in Experiments 3a and 3b, the decision attribution over the
final 300 trials, and the first-encounter tables of Experiments 4a and 4b.
The decision sources of Experiment 4 require both `rq4a` and `rq4b` in the manifest,
and Experiments 3a and 3b must use the same controller labels. Experiments are
processed in parallel (`--workers`, default 8), each holding its trial logs in
memory; lower `--workers` if memory is short. Runs with a different trial budget,
such as the 4800-trial runs, go into a separate report.

The figures of the thesis are drawn from the same runs with the same statistics,
in print layout:

```bash
python -m adaptive_sorting.analysis.thesis_figures \
  --manifest sources.json --extended-manifest sources_extended.json \
  --trace experiment_logs/rq2b/TRACE_ONA --tip-trace experiment_logs/rq2b/TRACE_TIP \
  --threshold-rq3a experiment_logs/rq3a/RUN/relational_ona/trials.csv \
  --threshold-rq3b experiment_logs/rq3b/RUN/relational_ona/trials.csv \
  --output-dir thesis_figures
```

- `sources.json` lists all five controllers and both ONA-TIP encodings in every
  experiment, and `sources_extended.json` lists ONA (relational) in the 4800-trial
  runs of `rq3a` and `rq3b`.
- `--trace` and `--tip-trace` are the output directories of the two implication
  traces, and `--threshold-rq3a` and `--threshold-rq3b` the trial logs of the
  suppression threshold intervention.
- Name figures to draw only those, for example `thesis_figures rq1 rq4`; each figure
  needs only its own inputs. `--help` lists the figure names.

A *stuck mapping* (Chapter 4 and Appendix B) is a color under one state of light
L1 in one run that is answered correctly in at most 10% of its presentations in
the final 300 trials. The stuck-mapping results of Section 4.3 and Appendices B.2
and B.3, including the table of the suppression threshold intervention, take the
same inputs except the traces:

```bash
python -m adaptive_sorting.analysis.stuck_mappings --manifest sources.json \
  --extended-manifest sources_extended.json \
  --threshold-rq3a experiment_logs/rq3a/RUN/relational_ona/trials.csv \
  --threshold-rq3b experiment_logs/rq3b/RUN/relational_ona/trials.csv
```

## Robot demo (ROS 2)

The ROS 2 backend runs a Panda arm on mock hardware with MoveIt 2 and shows the
scene in RViz. It implements the same execution interface as the no-op backend of
the quantitative experiments, so controllers and experiment code are unchanged.
The thesis figure of the RViz scenes shows each experiment run this way.

Install the dependencies of the ROS 2 package (after `rosdep init` and
`rosdep update`) and build the workspace:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths ros2/thesis_ws/src --ignore-src -y
./ros2/build-thesis-ws.sh
```

Start the backend for an experiment and keep it running. The scripts source ROS 2
and the workspace themselves:

```bash
./ros2/start-sorting-backend.sh rq3a
```

In a second terminal, run the experiment through the robot. The client runs in the
`thesis` Conda environment:

```bash
./ros2/run-sorting-experiment.sh rq3a --agent relational_ona
```

The client uses the trial budgets of the quantitative experiments. Replace `rq3a`
and the agent ID to run another experiment or controller, and start the backend
with the same experiment ID, which sets the number of bins and indicator lights.
Every trial executes a pick–place–return sequence, so a full run takes hours.
Phase-length options such as `--before-trials` and `--mixed-trials` shorten a run;
run `./ros2/run-sorting-experiment.sh <experiment> --help` for the options of each
experiment.

To execute a single action without a controller:

```bash
./ros2/run-sorting-action.sh --color blue --action place_to_bin5 --light-1 on
```

## License

The code of this repository is released under the [MIT License](LICENSE). ONA in
`third_party/OpenNARS-for-Applications/` keeps its own MIT License.
