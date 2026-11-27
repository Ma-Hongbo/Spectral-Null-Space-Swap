"""Export the official GSM8K main test split as benchmark JSONL."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/GSM8K/test.jsonl"
ANSWER_INSTRUCTION = r" Please put your final numeric answer in \boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", help="Optional Hugging Face dataset commit")
    args = parser.parse_args()

    from datasets import load_dataset

    examples = load_dataset("openai/gsm8k", "main", split="test", revision=args.revision)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        with temporary.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(examples):
                question = example["question"]
                row = {
                    "example_id": index,
                    "question": question,
                    "answer": example["answer"].rsplit("####", 1)[-1].strip(),
                    "messages": [{"role": "user", "content": question + ANSWER_INSTRUCTION}],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)
    print(f"Wrote {len(examples)} examples to {output}")


if __name__ == "__main__":
    main()
