"""Compare saved events against manually reviewed reference annotations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ez_worker.analytics.event_evaluation import evaluate_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0.5)
    parser.add_argument("--match-actors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_events(json.loads(args.events.read_text(encoding="utf-8")),
                             json.loads(args.reference.read_text(encoding="utf-8")),
                             args.tolerance_seconds, args.match_actors)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
