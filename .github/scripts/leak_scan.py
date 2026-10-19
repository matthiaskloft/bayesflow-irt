"""Fail if a pull request adds lines that reference private material.

Scans only the lines a PR adds (``git diff base...HEAD``). Generic patterns
live here; extra private terms come from the ``LEAK_TERMS`` environment
variable (a regex, set from a repository secret) so they are not published;
the scan fails if it is unset, since fork PRs receive no secrets.

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


def _diff_path(header: str) -> str | None:
    """Return the new-file path of a ``+++`` header, decoding Git's C quoting."""
    if header.startswith('"') and header.endswith('"'):
        # unicode_escape reads bytes as Latin-1, so literal UTF-8 bytes and
        # octal escapes both come back as single code points 0-255.
        raw = header[1:-1].encode("utf-8")
        header = raw.decode("unicode_escape").encode("latin-1").decode("utf-8")
    return header[2:] if header.startswith("b/") else None


def main() -> int:
    base = sys.argv[1]
    patterns = list(GENERIC)
    extra = os.environ.get("LEAK_TERMS", "").strip()
    if not extra:
        print("LEAK_TERMS is not set: the private-term check could not run.")
        return 1
    patterns.append(extra)
    regex = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

    diff = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--unified=0",
         f"{base}...HEAD"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="replace")

    # File headers only appear between "diff --git" and the first hunk, so an
    # added line that starts with "++ " is never mistaken for one.
    hits, path, line_no, in_header = [], None, 0, False
    # Split on "\n" only: splitlines() also breaks at U+2028 and similar
    # characters, which would hide the rest of an added line from the scan.
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            in_header, path = True, None
        elif in_header and line.startswith("+++ "):
            path = _diff_path(line[4:])
        elif line.startswith("@@"):
            in_header = False
            line_no = int(re.search(r"\+(\d+)", line).group(1))
        elif in_header:
            continue
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
