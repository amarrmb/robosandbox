def test_public_api_exposes_essentials():
    from robosandbox.eval_log import (
        EvalLogStore, EvalLogQuery,
        PolicyRow, DemoRow, EvalRow, EvalRunRow,
        SCHEMA_VERSION, walk_lineage,
        fit_training_dist_summary, score_trial,
        quantile_buckets, bucket_label,
    )
    assert SCHEMA_VERSION == 1
