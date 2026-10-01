#!/usr/bin/env python3
"""CI-API-backed facts for vibe-checks.yaml (GitHub REST; no gh CLI needed).

The goal-delivery graph reads pr_open / ci / reviews / main_ci from the CI API,
not from logs. This script is the command those checks run: it queries the
public GitHub REST API for the current commit and reports facts as exit codes.

Stdlib only. Exit codes: 0 = fact established, 1 = fact not established,
2 = stable error code (printed; never a raw exception or path).
"""

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TIMEOUT_SECONDS = 15
OK_CONCLUSIONS = {"success", "neutral", "skipped"}
BOT_NAMES = ("coderabbit", "cursor", "bugbot")


def _fail(code):
    print("ci_api_check: " + code)
    raise SystemExit(2)


def git(*args):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        _fail("E_GIT")
    return result.stdout.strip()


def repo_slug():
    tail = git("remote", "get-url", "origin").replace(":", "/").rstrip("/")
    tail = tail.removesuffix(".git")
    slug = "/".join(tail.split("/")[-2:])
    if not REPO_RE.match(slug):
        _fail("E_REPO_SLUG")
    return slug


def get(path):
    request = urllib.request.Request(
        API + path,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "vibe-agent-checks",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            _fail("E_RATE_LIMIT")
        _fail("E_API_HTTP_%d" % exc.code)
    except OSError:
        _fail("E_API_UNREACHABLE")


def bucket_of(entry):
    """Map one check-run or commit status to a gh pr checks bucket."""
    if "conclusion" in entry:  # check-run
        if entry.get("status") != "completed":
            return "pending"
        conclusion = entry.get("conclusion") or ""
        if conclusion in ("success", "neutral"):
            return "pass"
        if conclusion == "skipped":
            return "skipping"
        if conclusion == "cancelled":
            return "cancel"
        return "fail"
    state = entry.get("state") or ""  # commit status
    return {"success": "pass", "pending": "pending", "error": "cancel"}.get(
        state, "fail"
    )


def facts(repo, sha):
    """Blocking names for one commit, read from the checks and statuses APIs."""
    runs = get("/repos/%s/commits/%s/check-runs" % (repo, sha)).get("check_runs", [])
    statuses = get("/repos/%s/commits/%s/status" % (repo, sha)).get("statuses", [])
    names = []
    for entry in runs + statuses:
        name = entry.get("name") or entry.get("context") or "?"
        if bucket_of(entry) not in ("pass", "skipping"):
            names.append("%s:%s" % (name, bucket_of(entry)))
    return names, len(runs) + len(statuses)


def all_entries(repo, sha):
    runs = get("/repos/%s/commits/%s/check-runs" % (repo, sha)).get("check_runs", [])
    statuses = get("/repos/%s/commits/%s/status" % (repo, sha)).get("statuses", [])
    out = []
    for entry in runs + statuses:
        out.append(
            {
                "name": entry.get("name") or entry.get("context") or "?",
                "bucket": bucket_of(entry),
            }
        )
    return out


def report(label, blocking, total):
    if blocking:
        print("ci_api_check: FAIL %s blocking=%s" % (label, ",".join(blocking)))
        return 1
    print("ci_api_check: ok %s (checked %d)" % (label, total))
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ("pr-open", "ci", "buckets", "main-ci"):
        print("usage: ci_api_check.py pr-open|ci|buckets|main-ci")
        return 2
    command = argv[1]
    repo = repo_slug()

    if command == "pr-open":
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        head = git("rev-parse", "HEAD")
        remote = git("ls-remote", "--heads", "origin", "refs/heads/" + branch)
        if not remote or remote.split()[0] != head:
            print("ci_api_check: FAIL pr_open branch_not_pushed")
            return 1
        for pull in get("/repos/%s/pulls?state=open&per_page=100" % repo):
            if pull.get("head", {}).get("sha") == head or pull.get("head", {}).get(
                "ref"
            ) == branch:
                print("ci_api_check: ok pr_open (#%s)" % pull.get("number"))
                return 0
        print("ci_api_check: FAIL pr_open no_open_pr")
        return 1

    if command == "buckets":
        entries = all_entries(repo, git("rev-parse", "HEAD"))
        print(json.dumps(entries))
        return 0

    if command == "ci":
        blocking, total = facts(repo, git("rev-parse", "HEAD"))
        return report("ci", blocking, total)

    default = get("/repos/" + repo).get("default_branch") or "main"
    sha = get("/repos/%s/commits/%s" % (repo, default)).get("sha") or ""
    if not sha:
        _fail("E_DEFAULT_SHA")
    blocking, total = facts(repo, sha)
    return report("main_ci", blocking, total)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
