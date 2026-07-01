import argparse
import csv
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_COUNT = 10

PAIRS = [
    # (original_txt,               back-translation_txt,                   label)
    ("fineweb2_swe_matched.txt",  "fineweb2_swe_matched-en-sv.txt",       "swe-opus"),
    ("fineweb2_swe_matched.txt",  "fineweb2_swe_matched-gemma-en-sv.txt", "swe-gemma"),
    ("fineweb2_dan_10m.txt",      "fineweb2_dan_10m-en-da.txt",           "dan-opus"),
    ("fineweb2_dan_10m.txt",      "fineweb2_dan_10m-gemma-en-da.txt",     "dan-gemma"),
    ("fineweb2_isl_10m.txt",      "fineweb2_isl_10m-en-is.txt",           "isl-opus"),
    ("fineweb2_isl_10m.txt",      "fineweb2_isl_10m-gemma-en-is.txt",     "isl-gemma"),
    ("fineweb2_nob_10m.txt",      "fineweb2_nob_10m-en-nob.txt",          "nob-opus"),
    ("fineweb2_nob_10m.txt",      "fineweb2_nob_10m-gemma-en-nb.txt",     "nob-gemma"),
]

FIELDS = ["word", "count_original", "count_bt", "relfreq_original", "relfreq_bt", "ratio"]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def read_txt(path: Path) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
    return lines


def count_corpus(lines: list[str]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for line in lines:
        tokens = tokenize(line)
        counts.update(tokens)
        total += len(tokens)
    return counts, total


def write_candidates(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def compare(
    orig_counts: Counter,
    orig_total: int,
    bt_counts: Counter,
    bt_total: int,
    label: str,
    out_dir: Path,
) -> None:
    vocab = {
        w for w in orig_counts
        if orig_counts[w] >= MIN_COUNT and bt_counts.get(w, 0) >= MIN_COUNT
    }
    print(f"  shared vocab (≥{MIN_COUNT} in both): {len(vocab):,}")

    rows = []
    for word in vocab:
        co = orig_counts[word]
        cb = bt_counts[word]
        rf_o = co / orig_total
        rf_b = cb / bt_total
        rows.append({
            "word": word,
            "count_original": co,
            "count_bt": cb,
            "relfreq_original": round(rf_o, 8),
            "relfreq_bt": round(rf_b, 8),
            "ratio": round(rf_o / rf_b, 6),
        })

    over_orig = sorted(rows, key=lambda r: r["ratio"], reverse=True)
    over_bt   = sorted(rows, key=lambda r: r["ratio"])

    p1 = out_dir / f"candidates_{label}-over_original.csv"
    p2 = out_dir / f"candidates_{label}-over_bt.csv"
    write_candidates(over_orig[:500], p1)
    write_candidates(over_bt[:500],   p2)
    print(f"  → {p1.name}")
    print(f"  → {p2.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None)
    args = parser.parse_args()

    base = Path(args.dir) if args.dir else Path.cwd()

    orig_cache: dict[str, tuple[Counter, int]] = {}

    for orig_name, bt_name, label in PAIRS:
        orig_path = base / orig_name
        bt_path   = base / bt_name

        if not orig_path.exists():
            print(f"\n[skip] {label}: {orig_name} not found")
            continue
        if not bt_path.exists():
            print(f"\n[skip] {label}: {bt_name} not found")
            continue

        print(f"\n── {label} {'─' * (50 - len(label))}")

        if orig_name not in orig_cache:
            print(f"  Reading original: {orig_name} …", flush=True)
            orig_counts, orig_total = count_corpus(read_txt(orig_path))
            orig_cache[orig_name] = (orig_counts, orig_total)
            print(f"  tokens: {orig_total:,}  types: {len(orig_counts):,}")
        else:
            orig_counts, orig_total = orig_cache[orig_name]
            print(f"  Original: {orig_name} (cached)")

        print(f"  Reading back-translation: {bt_name} …", flush=True)
        bt_counts, bt_total = count_corpus(read_txt(bt_path))
        print(f"  tokens: {bt_total:,}  types: {len(bt_counts):,}")

        compare(orig_counts, orig_total, bt_counts, bt_total, label, base)


if __name__ == "__main__":
    main()
