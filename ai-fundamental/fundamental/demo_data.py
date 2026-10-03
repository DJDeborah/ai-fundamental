"""Synthetic text solely for pipeline smoke tests."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for i in range(1500):
            x, y = (i * 37) % 97, (i * 53) % 89
            text = f"Example {i}. Calculate {x} plus {y}. The result is {x+y}. This is a toy corpus."
            f.write(json.dumps({"text": text, "source": f"synthetic:{i}", "license": "CC0"}) + "\n")


if __name__ == "__main__":
    main()
