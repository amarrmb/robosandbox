import json
from robosandbox.eval_log.schema import (
    PolicyRow, DemoRow, EvalRow, EvalRunRow, SCHEMA_VERSION,
)


def test_policy_row_to_dict_roundtrip():
    row = PolicyRow(
        policy_id="reach_wide_30M",
        kind="ppo_neural",
        parent_policy_id=None,
        lineage_op=None,
        training_demo_set_id=None,
        training_steps=40_000_000,
        training_dist_summary=None,
        trained_at="2026-05-01T20:30:00Z",
        checkpoint_path="outputs/reach_wide",
        metadata={"obs_dim": 25, "act_dim": 4, "n_params": 74001},
    )
    d = row.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["policy_id"] == "reach_wide_30M"
    json.dumps(d)


def test_eval_row_basic():
    row = EvalRow(
        eval_id="eval_001",
        trial_id="00257",
        policy_id="reach_wide_30M",
        task_id="reach_target_franka",
        task_variant_hash="b8e7d2",
        sim_backend="newton",
        trial_seed=258,
        slice_axes={"target_xy_bucket": "front_right", "ood_score": 0.12, "is_ood": False},
        outcome={
            "success": True, "ever_within_threshold": True, "end_within_threshold": True,
            "min_dist_m": 0.024, "final_dist_m": 0.030,
            "steps_used": 128, "step_budget": 128,
        },
        wall_seconds=0.083,
        compute_resource="dgx-spark cuda:0",
        started_at="2026-05-01T20:30:14Z",
    )
    d = row.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["outcome"]["success"] is True
    json.dumps(d)


def test_demo_row_basic():
    row = DemoRow(
        demo_id="demo_47",
        demo_set_id="franka_picks_2026_03",
        task_id="pick_cube_franka_random",
        task_variant_hash="a3f2c1",
        operator="scripted_oracle",
        recorded_at="2026-03-01T14:25:33Z",
        sim_backend="mujoco",
        randomize_seed=47,
        duration_steps=612,
        outcome_label="success",
        demo_path="demos/franka_picks_2026_03/ep_47/",
    )
    json.dumps(row.to_dict())


def test_eval_run_row_basic():
    row = EvalRunRow(
        eval_id="eval_001",
        policy_id="reach_wide_30M",
        task_id="reach_target_franka",
        n_trials=1024,
        started_at="2026-05-01T20:30:00Z",
        completed_at="2026-05-01T20:31:25Z",
        git_sha="a9e6341",
        config_hash="fc73b9",
        compute_resource="dgx-spark cuda:0",
        command_line="robo-sandbox eval --task reach_target_franka --policy outputs/reach_wide",
    )
    json.dumps(row.to_dict())
