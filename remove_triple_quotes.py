import pathlib
import re
import sys


def main() -> None:
    root = pathlib.Path(".")
    pattern = re.compile(r'(|\'\'\'[\s\S]*?\'\'\')', re.MULTILINE)

    for path in root.rglob("*.py"):
        if any(part == "__pycache__" for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skip {path}: read fail {e}", file=sys.stderr)
            continue

        new_text = pattern.sub("", text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"Removed triple-quoted strings: {path}")


if __name__ == "__main__":
    main()


