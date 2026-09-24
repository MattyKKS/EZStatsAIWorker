"""Export timestamped source frames for independent event review, without models."""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step <= 0 or args.start < 0 or args.end <= args.start:
        parser.error("Use 0 <= start < end and a positive step")
    if args.output.exists():
        parser.error("Output already exists")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError("Cannot open source video")
    tiles = []
    try:
        for i in range(min(36, int((args.end - args.start) / args.step) + 1)):
            frame_index = round((args.start + i * args.step) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, round(640 * frame.shape[0] / frame.shape[1])))
            frame = cv2.copyMakeBorder(frame, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(frame, f"{frame_index / fps:.3f}s  frame {frame_index}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(frame)
        if not tiles:
            raise ValueError("No frames decoded")
        while len(tiles) % 3:
            tiles.append(tiles[0] * 0)
        sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), sheet):
            raise RuntimeError("Failed to write review sheet")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
