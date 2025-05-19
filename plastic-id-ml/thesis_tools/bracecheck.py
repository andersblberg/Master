"""
bracecheck.py  –  report unmatched { … } in LaTeX.

Yes, I messed up...

 - Ignores % comments.
 - Ignores escaped braces (\{ \}).
 - Works on a single file or on stdin.
"""

import sys, pathlib, re


def find_unmatched(text: str):
    stack, issues = [], []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("%", 1)[0]  # drop comments
        i = 0
        while i < len(line):
            if line[i] == "\\":  # skip escapes like \{
                i += 2
            elif line[i] == "{":
                stack.append((ln, i + 1))
                i += 1
            elif line[i] == "}":
                if stack:
                    stack.pop()
                else:
                    issues.append((ln, i + 1, "extra ‘}’"))
                i += 1
            else:
                i += 1

    issues.extend((ln, col, "missing ‘}’") for ln, col in stack)
    return issues


def main(path):
    text = pathlib.Path(path).read_text(encoding="utf8")
    problems = find_unmatched(text)
    if not problems:
        print("✓  All braces balanced.")
    else:
        for ln, col, kind in problems:
            print(f"{path}:{ln}:{col}: {kind}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bracecheck.py <file.tex>")
        sys.exit(1)
    main(sys.argv[1])
