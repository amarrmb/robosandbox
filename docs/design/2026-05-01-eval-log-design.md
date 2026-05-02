# Eval Log: Slice-aware, lineage-aware result store for IM+RL iteration

**Date:** 2026-05-01
**Scope:** internal tool for fast iteration on RoboSandbox itself (CLI-driven, single-machine, no multi-tenancy)
**Status:** Design — pending implementation plan

---

## 1. Goal

Make IM+RL iteration on RoboSandbox tractable by giving every eval first-class structure: per-trial outcomes, slice axes, and a lineage graph from base policy through fine-tunes. The system answers questions today's tooling can't: *"did this fine-tune improve the slice I cared about, or just churn the easy slice?"* — *"which checkpoint along the BC→PPO arc was the best overall?"* — *"if I add 5 demos in the failure zone, does the policy get better there without regressing elsewhere?"*

The thesis: in the IM+RL workflow that production robotics teams actually use, the bottleneck isn't algorithms — it's the eval-and-iterate substrate. RoboSandbox should be that substrate.

## 2. Hypothesis & alignment with 2026 research

This design is **on-thesis with where robotics research went in 2024–2026**, not behind it and not ahead of it:

- **IM+RL hybrids are the dominant production pattern** (Physical Intelligence, 1X, Skild, Gemini-Robotics 2025). Pure-from-scratch RL has essentially no production presence.
- **Scalar success rates are insufficient** — Stanford HAI's HEAP, LIBERO/SimplerEnv 2025 explicitly added per-trial breakdown and refused to publish single-number leaderboards.
- **Reward-design is the wall, not the algorithm** — NVIDIA Eureka, EUREKA-2, DeepMind GROOT/Gemini-Robotics tech reports.
- **Policy lineage is load-bearing** — 2025 Robot Learning Reproducibility Report (Berkeley/CMU) found 70% of published policies couldn't be reproduced because training lineage wasn't tracked.

Refinements from 2025 research that we fold in (without overreaching):

1. **OOD detection as a first-class slice axis** — silent OOD failures are the most dangerous failure mode (Pi-zero generalization studies, GR00T deployment retrospectives).
2. **Per-demo provenance, not per-set** — *which* demos predict *where* the policy works.
3. **Trial budget per slice** — distinguish "robust" from "lucky" (LeRobot 2026, ARC-AGI eval discipline).
4. **Sim-to-real as a pluggable comparison axis** — Habitat 2025, Newton-Industrial.
5. **Foundation-model fine-tune lineage** — RT-2, π0.5, OpenVLA workflows; lineage extends to HuggingFace Hub identifiers, not just local checkpoints.

## 3. Architecture

**Append-only event log + DuckDB views.** Every eval emits structured rows to JSONL files under `runs/eval_log/`. Reads via DuckDB SQL (one process, embedded, no DB server). The data shape grows over time without migrations: schema evolution is by additive columns + `schema_version` per row.

Trade-off picked over (B) embedded SQLite + ORM and (C) per-eval-directory + thin index:
- vs. SQLite: append-only matches RL's data flow naturally; no migrations as we evolve; no single-writer constraint; easy cloud sync later.
- vs. file-per-directory: scales past ~1000 runs cleanly; supports lineage graph queries; doesn't re-implement what DuckDB gives for free.

## 4. Data model

Four entity types, each an append-only JSONL file under `runs/eval_log/`. Every row carries `schema_version: 1`.

### 4.1 `policies.jsonl` — every checkpoint, with lineage

```json
{
  "schema_version": 1,
  "policy_id": "act_50k_distilled_v2",
  "kind": "ppo_neural | replay_trajectory | lerobot_act | lerobot_diffusion | foundation_fine_tune",
  "parent_policy_id": "act_50k_ckpt",                  // null for roots
  "lineage_op": "distill | warm_start_ppo | dagger | foundation_fine_tune",
  "training_demo_set_id": "franka_picks_2026_03",       // null for from-scratch RL
  "training_steps": 50000,
  "training_dist_summary": {                            // for OOD scoring (Mahalanobis fit)
    "mean": [...], "cov": [[...]], "n_samples": 1234
  },
  "trained_at": "2026-04-22T14:30:00Z",
  "checkpoint_path": "outputs/act_50k_distilled_v2",
  "metadata": { "obs_dim": 22, "act_dim": 8, "n_params": 74001 }
}
```

The `parent_policy_id` chain is what makes ACT → distill → PPO queryable as a lineage. Append-only: superseded versions stay in the log. `parent_policy_id` may be a local `policy_id` or an `hf://` URL for HuggingFace foundation models.

### 4.2 `demos.jsonl` — per-demo provenance

```json
{
  "schema_version": 1,
  "demo_id": "demo_20260301_142533_franka_picks_47",
  "demo_set_id": "franka_picks_2026_03",
  "task_id": "pick_cube_franka_random",
  "task_variant_hash": "a3f2c1...",                    // hash of task YAML at record time
  "operator": "scripted_oracle | <human_id>",
  "recorded_at": "2026-03-01T14:25:33Z",
  "sim_backend": "mujoco",
  "randomize_seed": 47,
  "duration_steps": 612,
  "outcome_label": "success | failure | partial",
  "demo_path": "demos/franka_picks_2026_03/ep_47/"
}
```

Demo-level (not just demo-set-level) provenance lets us answer *"did adding these specific 5 demos shift the failure slice?"*

### 4.3 `evals.jsonl` — one row per (policy × task × trial)

The hot table. Append one row per trial (1024 worlds × 1 eval = 1024 rows).

```json
{
  "schema_version": 1,
  "eval_id": "eval_20260501_2030_reach_wide_001",
  "trial_id": "00257",
  "policy_id": "reach_wide_30M",
  "task_id": "reach_target_franka",
  "task_variant_hash": "b8e7d2...",
  "sim_backend": "newton",                              // 'mujoco' | 'newton' | 'mujoco_vec' | 'real_so101' | ...
  "trial_seed": 258,
  "slice_axes": {
    "target_xy_bucket": "front_right",                   // bucketed at query time, raw stored
    "target_distance_from_home_m": 0.31,
    "ood_score": 0.12,
    "is_ood": false
  },
  "outcome": {
    "success": true,
    "ever_within_threshold": true,
    "end_within_threshold": true,
    "min_dist_m": 0.024,
    "final_dist_m": 0.030,
    "steps_used": 128,
    "step_budget": 128
  },
  "wall_seconds": 0.083,
  "compute_resource": "dgx-spark cuda:0",
  "started_at": "2026-05-01T20:30:14Z"
}
```

Slice axes are open-ended (free-form dict). Two reserved keys carry the OOD signal: `ood_score` and `is_ood`. Continuous values (e.g., `target_distance_from_home_m`) stored raw; bucketization happens at query time.

### 4.4 `eval_runs.jsonl` — group-level batch metadata

```json
{
  "schema_version": 1,
  "eval_id": "eval_20260501_2030_reach_wide_001",
  "policy_id": "reach_wide_30M",
  "task_id": "reach_target_franka",
  "n_trials": 1024,
  "started_at": "2026-05-01T20:30:00Z",
  "completed_at": "2026-05-01T20:31:25Z",
  "git_sha": "a9e6341...",
  "config_hash": "fc73b9...",
  "compute_resource": "dgx-spark cuda:0",
  "command_line": "robo-sandbox eval --task reach_target_franka --policy ..."
}
```

One row per eval batch (vs `evals.jsonl` which is one row per trial). Reproducibility: `git_sha` + `command_line` + `config_hash` together let any past result be re-run exactly.

## 5. Active behaviors

Two things the system performs automatically; everything else is passive storage and querying.

### 5.1 Lineage propagation

`parent_policy_id` is set at policy *creation*, not at eval time. Three creation paths:

- **PPO warm-start.** `train_ppo` reads parent's `policy.json`, extracts `policy_id`, writes new policy with `parent_policy_id` set. ~10 LOC change in `NeuralPolicy.save()`.
- **Distillation.** `scripts/distill_act_to_mlp.py` records source ACT checkpoint as parent + `lineage_op = "distill"`. ~5 LOC.
- **Foundation-model fine-tune.** LeRobot adapter records HF identifier (e.g., `hf://lerobot/act_aloha`) as parent. ~15 LOC.

Pre-existing checkpoints become roots (`parent_policy_id = null`). No retroactive reconstruction. New work tracks lineage forward.

The `lineage` query walks the chain by following `parent_policy_id` recursively. Cycle detection: max-depth 32; error past that.

### 5.2 OOD score computation

Answers: *"for this trial's task condition, how far is it from the training distribution of the policy?"*

For policies with a recorded training demo set (ACT, distill, BC):

1. At policy registration, compute `training_dist_summary` — mean and covariance over task-randomization parameters in the demo set.
2. Store summary in `policies.jsonl`.
3. At eval time, compute Mahalanobis distance of each trial's task conditions to the summary; normalize to [0, 1] by capping at the 99th percentile of training-distribution distances.
4. `is_ood = ood_score > 0.5` (configurable per task).

For from-scratch RL: sample 1000 conditions from the YAML's `randomize` block at policy registration; use as summary.

For policies with neither (random-init, replay-trajectory): `ood_score = null`, `is_ood = null`.

**Mahalanobis assumes roughly Gaussian distributions.** For uniform `xy_jitter` this works fine. For multimodal future cases we'll extend to KDE; the schema accommodates the change (just swap the score function).

**Insufficient samples guard:** if training distribution has <30 samples, emit `ood_score = null`. CLI displays *"OOD: insufficient samples"* rather than a misleading number.

## 6. Slice axes — the YAML extension

**Naming note:** the term `slice_axes` appears in two places with related-but-distinct roles:
- In **task YAML** (this section) it is the *configuration* — which axes the system should compute per trial.
- In **`evals.jsonl`** (section 4.3, `slice_axes` field) it is the *per-trial computed values* derived from that configuration.

Optional `slice_axes` block in task YAML. Default behavior (block absent): infer slice axes from the `randomize` block.

```yaml
slice_axes:
  - { name: target_xy, source: object_pose, object: target_marker, fields: [x, y] }
  - { name: target_distance_from_home, source: derived,
      expr: "distance(target_marker.xyz, robot.home_ee_xyz)" }
```

Loader (`tasks/loader.py`) parses into `task.slice_axes`. Backward-compat: existing tasks without the block work unchanged.

Continuous slice values are **bucketized at query time**, not at write time. Default: 3-quantile (terciles) per axis. Override via `--buckets` flag.

## 7. CLI surface

### 7.1 Write path

The existing `robo-sandbox eval` command extends to also append rows to the eval log. No new flags. Backwards-compatible: legacy `result.json` keeps being written too.

The recorder's `LocalRecorder.end_episode()` extends to also append a row to `demos.jsonl`. ~10 LOC.

### 7.2 New top-level: `robo-sandbox results`

```bash
# Overview
robo-sandbox results list                                        # policies w/ headline stats, last 7d default

# Slice breakdown
robo-sandbox results slice <policy_id> --axis target_xy_bucket --axis is_ood

# Lineage
robo-sandbox results lineage <policy_id>                         # walks parent chain

# Side-by-side comparison with Wilson significance test (default on)
robo-sandbox results compare <policy_a> <policy_b> [--task <task>] [--by <axis>]

# Regression hunt vs lineage parent
robo-sandbox results regression <policy_id>

# Reproducibility: print git_sha + exact command
robo-sandbox results repro <eval_id>

# SQL escape hatch — DuckDB over the JSONL views
robo-sandbox results sql 'SELECT ... FROM evals WHERE policy_id = ...'

# Visual: workspace heatmap (opens viewer at /results/heatmap/<eval_id>)
robo-sandbox results heatmap <policy_id> --axis target_x --axis target_y

# One-shot migration of legacy runs/ directories (opt-in)
robo-sandbox results migrate runs/
```

The SQL escape hatch is a deliberate choice: useful for one-off questions; risk is that downstream scripts depend on raw schema. Acceptable risk given the appeal of unblocking power users.

## 8. Module structure

Self-contained package; passive data layer (does not render videos, talk to sim, train models).

```
robosandbox/eval_log/
├── __init__.py          (exports: append_eval, append_policy, append_demo, query)
├── schema.py            (dataclasses: PolicyRow, EvalRow, DemoRow, EvalRunRow + JSON dump)
├── store.py             (low-level append; opens file, writes line, flushes, closes)
├── ood.py               (Mahalanobis fit/score; pluggable for future KDE)
├── lineage.py           (chain walker; cycle detection; serialization)
├── slicing.py           (bucketization for continuous slice values)
└── query.py             (DuckDB connection, view registration, canned queries)
```

Single new external dependency: `duckdb>=0.10`.

## 9. Wiring — minimal, additive changes

| File | Change | LOC |
|---|---|---|
| `cli.py` (`_eval_parallel_cli`, `_eval_cli`) | Append eval_log rows after existing result.json write | ~30 |
| `rl/ppo.py` (`NeuralPolicy.save`) | Record `policy_id` + `parent_policy_id` + `lineage_op` | ~15 |
| `recorder/local.py` (`end_episode`) | Append demo row | ~10 |
| `scripts/distill_act_to_mlp.py` | Record lineage on output checkpoint | ~5 |
| `policy/lerobot_adapter.py` | Record HF parent identifier | ~15 |
| `tasks/loader.py` | Parse optional `slice_axes` block | ~20 |
| `cli.py` (new `results` subparser group) | Wire to `eval_log.query` functions | ~150 |
| `viewer/server.py` + new HTML | `/results/heatmap/<eval_id>` page | ~80 + ~200 |

Total wiring: ~525 LOC across existing files. New code in `eval_log/` module: ~600 LOC. Migration script: ~50 LOC.

## 10. What's intentionally NOT in v1

- Active learning loop ("auto-suggest 5 demos for failing slice")
- Web dashboard (CLI + single viewer page is enough)
- KDE / non-Gaussian OOD (Mahalanobis only)
- Semantic OOD (vision feature distance) — task-condition OOD only
- Multi-policy ensembling at eval
- Cloud storage backend
- Notification / alerting on regressions
- Real-robot eval *recording* code (the `sim_backend = "real_<x>"` slot exists; no recording-from-real path ships)
- Multi-tenant auth

## 11. Backward compatibility

Three guarantees:

- Existing eval CLI flags **unchanged**. New behavior layered on top.
- Existing `result.json` format **kept**. Some scripts depend on it.
- Eval log absent or empty → all `results` commands print *"no data; run eval first"* and exit 0. No crashes on fresh checkouts.

## 12. Testing

**Unit tests (per module, fast):**
- `store.py`: append → read round-trip; concurrent writes; rotation when file > N MB
- `ood.py`: Mahalanobis on synthetic Gaussian → score 0 at mean, monotonic; degenerate covariance handled
- `lineage.py`: 5-deep traversal; cycle detection raises; orphan parent reference returns "unknown"
- `slicing.py`: edge cases (all same value → 1 bucket, empty → empty)
- `query.py`: each canned query against fixture log with known outputs

**Integration tests (slower, end-to-end):**
- `robo-sandbox eval` on tiny task (4 worlds, 50 steps) → assert eval_log row counts
- `robo-sandbox results compare` on two known checkpoints → assert delta sign + significance
- `robo-sandbox results regression` on deliberately-regressed lineage → assert flag fires
- Migration: build fixture `runs/old/`, run migrate, verify rows appear

**Smoke (CI-friendly, no GPU):**
MuJoCo single-world reach, 4 trials, 30 max-steps. Whole pipeline <30 sec.

Total new test code: ~600 LOC across the three layers.

## 13. Rollout — three phases over ~3 weeks

**Phase 1 (week 1): foundation.** Ship `eval_log/` module with full schema, store, OOD, lineage, slicing, query. `slice_axes` YAML extension (parser side). Migration script (opt-in). Tests passing. *Existing CLI behavior unchanged.*

**Phase 2 (week 2): wire eval + train.** `_eval_*_cli` start emitting eval_log rows alongside `result.json`. `NeuralPolicy.save()` records lineage. `LocalRecorder.end_episode()` records demos. `distill_act_to_mlp.py` records lineage. *Every new run automatically populates the log.*

**Phase 3 (week 3): CLI + viewer.** `results` subcommand group (8 commands). Viewer heatmap page. Three docs pages (concept, guide, CLI reference). *Full user-facing surface.*

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Schema lock-in: append-only is hard to migrate | Every row carries `schema_version`. The first ~2 weeks (Phase 2 + first half of Phase 3, when real eval/train runs first start populating the log) are an explicit dogfood window during which the schema may add or rename fields without bumping `schema_version`. After that, `schema_version` is incremented for any breaking field change and readers handle multiple versions. New non-breaking columns add freely. |
| DuckDB queries slow at >1M rows | Rotate `evals.jsonl` monthly (`evals_2026_05.jsonl` etc.). DuckDB reads globs. 614k rows/year is well under sweet spot. |
| OOD score misleading on early evals (covariance under-estimated with few samples) | Emit `null` if training distribution has <30 samples. CLI displays *"insufficient samples"*. |
| Lineage graph fragments because old policies have no IDs | Documented limitation. Migration script supports manual `--parent <id>` flag for hand-edits. |
| This is the wrong abstraction and we throw it away | The eval_log is *additive*. Existing `result.json` keeps working. Worst case: delete `runs/eval_log/` + new module. Nothing else breaks. |

## 15. Success criteria

The design is successful if, two weeks after Phase 3 ships:

1. I can answer *"did my latest PPO checkpoint improve on the failure slice from yesterday's run?"* with one CLI command in <5 seconds.
2. When I add 5 new demos and retrain, I can quantify whether the failure zone shrunk on the *exact* trials that were failing.
3. I never train a policy and not know what it's worse at than the previous policy.
4. The cube-pick experiment (when revisited with EE-space PPO) takes half the wall time we spent on reach, because we're not running blind.

The negative success criterion: if at the end of Phase 3 I'm not consistently using `results compare` and `results regression` in my own iteration loop, the abstraction was wrong and we should reconsider.
