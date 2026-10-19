"""Fail if a pull request adds lines that reference private material.

Scans only the lines a PR adds (``git diff base...HEAD``). Generic patterns
live here; extra private terms come from the ``LEAK_TERMS`` environment
variable (a regex, set from a repository secret) so they are not published.

Usage: python .github/scripts/leak_scan.py <base-ref>
"""

import os
import re
import subprocess
import sys

GENERIC = [
    r"\bEXP-?\d",  # experiment record IDs
    r"\bADR-?\d",  # decision record IDs
    r"bayesflow-irt-research",  # the private repo
    r"\bpc\d{5}\b",  # lab hostnames
    r"runpod",
    r"[A-Za-z]:\\Users\\",  # Windows home paths
    r"/home/\w+",  # Linux home paths
    r"/Users/\w+",  # macOS home paths
    r"\.claude/worktrees",
    r"zotero_linked_attachments|nexthessenbox",
]
SELF = ".github/scripts/leak_scan.py"


def main() -> int:
    base = sys.argv[1]
    patterns = list(GENERIC)
    extra = os.environ.get("LEAK_TERMS", "").strip()
    if extra:
        patterns.append(extra)
    regex = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

    diff = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True,
    ).stdout

    hits, path, line_no = [], None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            line_no = int(re.search(r"\+(\d+)", line).group(1))
        elif line.startswith("+") and path and path != SELF:
            if regex.search(line[1:]):
                # Do not echo the line: the match may itself be private.
                hits.append(f"{path}:{line_no}")
            line_no += 1

    if hits:
        print("Possible private references in added lines:")
        print("\n".join(f"  {h}" for h in hits))
        return 1
    print("Leak scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
