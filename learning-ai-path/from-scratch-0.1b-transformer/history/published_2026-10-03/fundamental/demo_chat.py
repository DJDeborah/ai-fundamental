"""A few memorization examples to show what SFT changes in a tiny model."""
import argparse
import json
from pathlib import Path


EXAMPLES = [
    ("hello", "Hi!"),
    ("what is your name?", "I am TinyBot."),
    ("what can you do?", "I demonstrate how training changes text generation."),
    ("2+2?", "4"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for question, answer in EXAMPLES:
            f.write(json.dumps({"prompt": f"User: {question}\nAssistant: ", "completion": answer}) + "\n")
    print(f"wrote {len(EXAMPLES)} toy examples to {a.out}")


if __name__ == "__main__":
    main()
