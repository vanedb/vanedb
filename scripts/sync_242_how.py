#!/usr/bin/env python3
"""Replace only issue 242's bounded How section; never reconstruct its body."""
import json
import re
import subprocess
import sys

REPO = "vanedb/vanedb"
# Existing issue's safety footer is an explicit legacy end boundary. Otherwise
# a following level 1/2 heading is required; EOF is deliberately not a boundary.
FOOTER = "**Do not** put `Fixes`/`Closes` next to `#198` / `#226` / this issue on harness PRs until both boxes above have evidence URLs. Shut the issue from the GitHub UI after evidence, not via merge keywords."


def replace_how(body, replacement):
    headings, boundaries = [], []
    offset = 0
    fence = None
    for line in body.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", text)
        if fence:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = None
        elif marker:
            fence = marker[1]
        else:
            # A Setext heading can start in a preceding multi-line paragraph.
            # Do not guess its boundary (or treat it as part of How).
            if re.match(r"^ {0,3}(?:=+|-+)\s*$", text):
                raise ValueError("ambiguous Setext heading/rule; use explicit ## headings")
            heading = re.match(r"^ {0,3}(#{1,2})(?:[ \t]+(.*)|[ \t]*)$", text)
            if heading:
                boundaries.append(offset)
                # ATX closing hashes need preceding whitespace. `How#` and
                # `How###` are different headings, never the managed section.
                title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
                if heading[1] == "##" and title == "How":
                    headings.append((offset + len(line), line))
            elif text == FOOTER:
                boundaries.append(offset)
        offset += len(line)
    if fence or len(headings) != 1:
        raise ValueError("expected one unambiguous ## How heading and closed code fences")
    start, heading_line = headings[0]
    ends = [pos for pos in boundaries if pos >= start]
    if not ends:
        raise ValueError("How needs a following heading or the exact safety footer")
    if not replacement.startswith("## How\n"):
        raise ValueError("invalid How template")
    newline = "\r\n" if heading_line.endswith("\r\n") else "\n"
    content = replacement[len("## How\n"):].strip("\n")
    content = ("\n" + content + "\n\n").replace("\n", newline)
    return body[:start] + content + body[min(ends):]


def read_body():
    result = subprocess.run(
        ["gh", "issue", "view", "242", "--repo", REPO, "--json", "body"],
        check=True, capture_output=True,
    )
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise ValueError("issue response is not an object")
    body = document["body"]
    if not isinstance(body, str) or not body:
        raise ValueError("issue body is missing or unreadable")
    return body


def main():
    replacement = sys.stdin.read()
    original = read_body()
    updated = replace_how(original, replacement)
    if original == updated:
        print("#242 How already current; no write")
        return
    # Catch edits during preparation. GitHub's issue-edit API has no atomic
    # compare-and-swap; workflow concurrency serializes this automation only.
    if read_body() != original:
        raise ValueError("issue changed during preparation; retry from its new body")
    subprocess.run(
        ["gh", "issue", "edit", "242", "--repo", REPO, "--body-file", "-"],
        input=updated.encode("utf-8"), check=True,
    )
    print("updated #242 How; preserved all content outside the section")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"refused: {error}")
