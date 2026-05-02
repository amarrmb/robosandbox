"""NeuralPolicy.save() emits lineage to eval_log + policy.json."""
import json

from robosandbox.eval_log.store import POLICIES


def test_train_ppo_writes_lineage(tmp_path, monkeypatch):
    """When NeuralPolicy.save() is called with parent_policy_id set, a
    PolicyRow with parent linkage appears in the log + policy.json."""
    monkeypatch.chdir(tmp_path)
    from robosandbox.rl.obs_encoder import ObsEncoder
    from robosandbox.rl.ppo import ActorCritic, NeuralPolicy

    enc = ObsEncoder(["target_marker"], n_dof=7)
    ac = ActorCritic(enc.obs_dim, 4)
    policy = NeuralPolicy(
        ac, enc, delta_scale=0.05,
        action_space="ee_xyz", ee_delta_scale=0.025, ee_body="hand",
        ee_kine=None,
        policy_id="test_ppo_v1",
        parent_policy_id="test_act_root",
        lineage_op="warm_start_ppo",
    )
    save_dir = tmp_path / "outputs" / "test_ppo_v1"
    policy.save(save_dir)
    # Check policy.json includes lineage fields
    cfg = json.loads((save_dir / "policy.json").read_text())
    assert cfg["policy_id"] == "test_ppo_v1"
    assert cfg["parent_policy_id"] == "test_act_root"
    assert cfg["lineage_op"] == "warm_start_ppo"
    # Check eval_log saw it
    rows = (tmp_path / "runs" / "eval_log" / POLICIES).read_text().splitlines()
    assert any(json.loads(r)["policy_id"] == "test_ppo_v1" for r in rows)
    pol_row = next(
        json.loads(r) for r in rows if json.loads(r)["policy_id"] == "test_ppo_v1"
    )
    assert pol_row["parent_policy_id"] == "test_act_root"
    assert pol_row["lineage_op"] == "warm_start_ppo"
