"""One-to-one event evaluation within explicitly completed annotation windows."""
import math


def evaluate_events(predictions, reference, tolerance_seconds=0.5, match_actors=False):
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be finite and nonnegative")
    windows = reference.get("annotated_windows", [])
    if not windows or reference.get("annotation_status") != "reviewed":
        raise ValueError("Reference must contain reviewed, fully annotated windows")
    for start, end in windows:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid annotation window")
    if any(b[0] < a[1] for a, b in zip(sorted(windows), sorted(windows)[1:])):
        raise ValueError("Annotation windows must not overlap")

    def window_for(t):
        return next((i for i, (start, end) in enumerate(windows) if start <= t < end), None)

    truth = reference.get("events", [])
    event_types = reference.get("event_types")
    if event_types is not None and (not event_types or any(e["event_type"] not in event_types for e in truth)):
        raise ValueError("Reference events must belong to the declared event_types")
    for e in list(predictions) + truth:
        if not e.get("event_type") or not math.isfinite(e["time_seconds"]) or e["time_seconds"] < 0:
            raise ValueError("Each event needs a type and finite nonnegative time_seconds")
    if any(window_for(e["time_seconds"]) is None for e in truth):
        raise ValueError("Reference event is outside annotated windows")
    selected = [(i, e) for i, e in enumerate(predictions) if window_for(e["time_seconds"]) is not None
                and (event_types is None or e["event_type"] in event_types)]
    adjacency = {}
    for i, target in enumerate(truth):
        choices = []
        for j, pred in selected:
            if pred["event_type"] != target["event_type"]:
                continue
            if window_for(pred["time_seconds"]) != window_for(target["time_seconds"]):
                continue
            if match_actors and any(target.get(k) is None or pred.get(k) != target[k]
                                    for k in ("actor_track_id",)):
                continue
            if match_actors and target.get("target_track_id") is not None and pred.get("target_track_id") != target["target_track_id"]:
                continue
            error = abs(pred["time_seconds"] - target["time_seconds"])
            if error <= tolerance_seconds:
                choices.append((error, j))
        adjacency[i] = [j for _, j in sorted(choices)]

    # Augmenting paths maximize the number of one-to-one matches. Greedy nearest
    # matching can miss valid pairs when events happen close together.
    assigned = {}

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(truth)):
        augment(i, set())
    matched_truth = set(assigned.values())
    matches = [{"prediction_index": j, "reference_index": i,
                "timing_error_seconds": abs(predictions[j]["time_seconds"] - truth[i]["time_seconds"])}
               for j, i in sorted(assigned.items())]
    missed = [i for i in range(len(truth)) if i not in matched_truth]
    false = [i for i, _ in selected if i not in assigned]

    def metrics(tp, fp, fn):
        return dict(tp=tp, fp=fp, fn=fn,
                    precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)

    classes = sorted({e["event_type"] for _, e in selected} | {e["event_type"] for e in truth})
    per_class = {c: metrics(sum(truth[i]["event_type"] == c for i in matched_truth),
                            sum(predictions[i]["event_type"] == c for i in false),
                            sum(truth[i]["event_type"] == c for i in missed)) for c in classes}
    return dict(overall=metrics(len(matches), len(false), len(missed)), per_class=per_class,
                matches=matches, false_positive_indices=false, missed_reference_indices=missed,
                ignored_prediction_count=len(predictions) - len(selected),
                tolerance_seconds=tolerance_seconds, match_actors=match_actors,
                event_types=event_types,
                note="Scores apply only to reviewed windows, not the entire clip or all football accuracy")
