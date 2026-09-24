import pytest

from ez_worker.analytics.event_evaluation import evaluate_events


def event(t, kind="pass", **kwargs):
    return dict(time_seconds=t, event_type=kind, **kwargs)


def reference(events):
    return dict(annotation_status="reviewed", annotated_windows=[[0, 10]], events=events)


def test_duplicate_prediction_is_false_positive():
    score = evaluate_events([event(1), event(1.1)], reference([event(1)]))
    assert score["overall"]["tp"] == 1
    assert score["overall"]["fp"] == 1


def test_shot_instead_of_pass_is_not_correct():
    score = evaluate_events([event(1, "shot_attempt")], reference([event(1)]))
    assert score["overall"] == dict(tp=0, fp=1, fn=1, precision=0, recall=0, f1=0)


def test_unannotated_time_is_not_scored():
    score = evaluate_events([event(11)], reference([]))
    assert score["ignored_prediction_count"] == 1
    assert score["overall"]["precision"] is None


def test_matching_is_not_greedy():
    score = evaluate_events([event(1.3), event(.7)], reference([event(1), event(1.6)]), .4)
    assert score["overall"]["tp"] == 2


def test_wrong_actor_not_counted_in_identity_aware_mode():
    score = evaluate_events([event(1, actor_track_id=2)], reference([event(1, actor_track_id=1)]), match_actors=True)
    assert score["overall"]["tp"] == 0


def test_unreviewed_template_refused():
    with pytest.raises(ValueError, match="reviewed"):
        evaluate_events([], dict(annotation_status="draft", annotated_windows=[[0, 10]], events=[]))


def test_events_cannot_match_across_annotation_gap():
    ref = dict(annotation_status="reviewed", annotated_windows=[[0, 1], [1.1, 2]], events=[event(.9)])
    assert evaluate_events([event(1.2)], ref)["overall"]["tp"] == 0


def test_class_scope_is_explicit():
    ref = reference([event(1)])
    ref["event_types"] = ["pass"]
    score = evaluate_events([event(1), event(1, "touch")], ref)
    assert score["overall"]["fp"] == 0
    assert score["ignored_prediction_count"] == 1
