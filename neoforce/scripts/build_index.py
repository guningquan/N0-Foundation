"""Build an OpenNeoData sequence index for OpenNeoDataset.

Walks ``<root>/<robot>/meta/episodes/chunk-*/file-*.parquet`` once and writes a JSON
holding, per (episode, wrist side), the resolved mp4 paths and the episode's
``[t0, t1]`` window in each stream, so the dataset starts without re-scanning the
dataset metadata on every launch.

A wrist side is emitted only when all three of its streams exist: the wrist RGB view
and both tactile sensors. Sides missing any of them are skipped rather than filled in,
because a fabricated stream would become fabricated ground truth.

Only needed to index a different selection of episodes; the shipped
``splits/train.jsonl`` and ``splits/test.jsonl`` are ready to use as they are.

    python scripts/build_index.py \
      --root /path/to/OpenNeoData \
      --output splits/my_split.json

    # restrict to a set of <task_slug>-<robot> ids
    python scripts/build_index.py --root /path/to/OpenNeoData \
      --categories-json my_categories.json \
      --output splits/my_subset.json
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoforce.data.openneo import (  # noqa: E402
    TACTILE_SIDES,
    category_id,
    rgb_key,
    tactile_key,
)

EMBODIMENTS = ("aloha", "arx5", "arx5_single", "flexiv", "ur", "umi", "umi_single")
WRIST_SIDES = ("left", "right")


def load_categories(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("categories", list(raw.keys()))
    return {str(item["id"] if isinstance(item, dict) else item) for item in raw}


def video_path(root: Path, robot: str, key: str, chunk_index: int, file_index: int) -> Path:
    return root / robot / "videos" / key / f"chunk-{int(chunk_index):03d}" / f"file-{int(file_index):03d}.mp4"


def stream_columns(key: str) -> List[str]:
    return [
        f"videos/{key}/chunk_index",
        f"videos/{key}/file_index",
        f"videos/{key}/from_timestamp",
        f"videos/{key}/to_timestamp",
    ]


def build_robot(
    root: Path,
    robot: str,
    allowed: Optional[set],
    max_episodes_per_category: int,
    verify_paths: bool,
    stats: Counter,
) -> List[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "pyarrow is required to read the LeRobot episode metadata and is not installed "
            "in this interpreter. Install it with `pip install pyarrow`. Indexing is a "
            "one-time offline step, so it can also be run in a separate environment: the "
            "resulting index JSON is all that training and evaluation need."
        ) from exc

    info_path = root / robot / "meta" / "info.json"
    if not info_path.is_file():
        print(f"  [skip] {robot}: no meta/info.json")
        return []
    info = json.load(open(info_path, "r", encoding="utf-8"))
    features = info.get("features", {})
    fps = float(info.get("fps", 30))

    # Only wrist sides whose RGB + both tactile streams all exist in this embodiment.
    usable_sides = [
        side
        for side in WRIST_SIDES
        if rgb_key(side) in features
        and all(tactile_key(side, sensor) in features for sensor in TACTILE_SIDES)
    ]
    if not usable_sides:
        print(f"  [skip] {robot}: no wrist side has both a wrist view and two tactile streams")
        return []

    keys = []
    for side in usable_sides:
        keys.append(rgb_key(side))
        keys.extend(tactile_key(side, sensor) for sensor in TACTILE_SIDES)
    columns = ["episode_index", "tasks", "length"] + [c for key in keys for c in stream_columns(key)]

    parquets = sorted(glob.glob(str(root / robot / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    if not parquets:
        print(f"  [skip] {robot}: no meta/episodes parquet files")
        return []

    per_category: Counter = Counter()
    entries: List[Dict] = []
    for parquet in parquets:
        table = pq.read_table(parquet, columns=columns).to_pydict()
        for i in range(len(table["episode_index"])):
            task = table["tasks"][i]
            task = task[0] if isinstance(task, list) else task
            category = category_id(task, robot)
            if allowed is not None and category not in allowed:
                continue
            if max_episodes_per_category > 0 and per_category[category] >= max_episodes_per_category:
                continue

            length = int(table["length"][i])
            episode_index = int(table["episode_index"][i])
            emitted = False
            for side in usable_sides:
                streams = {}
                ok = True
                for label, key in [
                    ("rgb", rgb_key(side)),
                    ("left", tactile_key(side, "left")),
                    ("right", tactile_key(side, "right")),
                ]:
                    chunk_index = table[f"videos/{key}/chunk_index"][i]
                    file_index = table[f"videos/{key}/file_index"][i]
                    if chunk_index is None or file_index is None:
                        ok = False
                        break
                    path = video_path(root, robot, key, chunk_index, file_index)
                    if verify_paths and not path.is_file():
                        stats["missing_video"] += 1
                        ok = False
                        break
                    streams[label] = {
                        "path": str(path),
                        "t0": float(table[f"videos/{key}/from_timestamp"][i]),
                        "t1": float(table[f"videos/{key}/to_timestamp"][i]),
                    }
                if not ok:
                    stats["skipped_side"] += 1
                    continue
                entries.append(
                    {
                        "robot": robot,
                        "task": task,
                        "category": category,
                        "episode_index": episode_index,
                        "length": length,
                        "wrist_side": side,
                        "rgb": streams["rgb"],
                        "tactile": {"left": streams["left"], "right": streams["right"]},
                    }
                )
                emitted = True
            if emitted:
                per_category[category] += 1

    stats["fps_" + str(int(fps))] += 1
    print(f"  {robot}: {len(entries)} sequences over {len(per_category)} categories (sides: {usable_sides})")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OpenNeoData online-inference sequence index.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--robots", nargs="+", default=list(EMBODIMENTS))
    parser.add_argument("--categories-json", default=None, help="Restrict to these `<task_slug>-<robot>` ids.")
    parser.add_argument("--max-episodes-per-category", type=int, default=0, help="0 = no cap.")
    parser.add_argument(
        "--verify-paths",
        action="store_true",
        help="stat() every mp4 while indexing. Slower, but turns missing-file errors into "
        "index-time skips instead of training-time failures.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")
    allowed = load_categories(args.categories_json)
    if allowed is not None:
        print(f"restricting to {len(allowed)} categories from {args.categories_json}")

    stats: Counter = Counter()
    sequences: List[Dict] = []
    fps = 30.0
    for robot in args.robots:
        info_path = root / robot / "meta" / "info.json"
        if info_path.is_file():
            fps = float(json.load(open(info_path, "r", encoding="utf-8")).get("fps", fps))
        sequences.extend(
            build_robot(root, robot, allowed, args.max_episodes_per_category, args.verify_paths, stats)
        )

    if not sequences:
        raise SystemExit("No sequences were indexed; check --root, --robots and --categories-json.")

    payload = {
        "root": str(root),
        "fps": fps,
        "num_sequences": len(sequences),
        "robots": sorted({entry["robot"] for entry in sequences}),
        "categories": sorted({entry["category"] for entry in sequences}),
        "sequences": sequences,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"\nwrote {output}")
    print(f"  sequences : {len(sequences)}")
    print(f"  categories: {len(payload['categories'])}")
    print(f"  robots    : {', '.join(payload['robots'])}")
    print(f"  fps       : {fps}")
    if stats["skipped_side"]:
        print(f"  skipped wrist sides: {stats['skipped_side']} (missing stream entries)")
    if stats["missing_video"]:
        print(f"  missing mp4 files  : {stats['missing_video']}")


if __name__ == "__main__":
    main()
