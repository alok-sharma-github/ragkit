"""Create the GitHub -> AWS OIDC trust so CI deploys without long-lived keys.

WHY OIDC AND NOT AN ACCESS KEY IN A GITHUB SECRET. A stored key is a bearer
credential with no expiry: it works from anywhere, for anyone who reads it, until
someone remembers to rotate it. OIDC issues a token per workflow run, scoped to
this repository and valid for minutes.

THE `sub` CONDITION IS THE WHOLE SECURITY BOUNDARY, and the obvious value for it
is wrong. Every tutorial writes:

    "token.actions.githubusercontent.com:sub": "repo:owner/repo:*"

That trailing `*` matches `pull_request` too -- INCLUDING pull requests opened from
forks. So anyone on the internet could open a PR against this repo and have their
workflow assume a role that can deploy production. The wildcard reads like "this
repo" and means "this repo, and anyone who can address it".

Scoped here to two explicit refs instead. A fork PR does not match, and neither
does a new branch until someone deliberately adds it. Same rule as the demo
guard: deny by default, allow by explicit pattern.

IDEMPOTENT. Safe to re-run; existing provider/role/policy are updated rather than
duplicated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = "alok-sharma-github/ragkit"
ROLE = "ragkit-github-deploy"
POLICY = "ragkit-lightsail-deploy"
PROVIDER_HOST = "token.actions.githubusercontent.com"
# Branches allowed to deploy. Adding one here is a deliberate act.
ALLOWED_REFS = ("refs/heads/main", "refs/heads/product")


def aws(*args: str, check: bool = True) -> tuple[int, str]:
    r = subprocess.run(
        ["aws", *args], capture_output=True, text=True, cwd=ROOT,
        env={**_env(), "AWS_DEFAULT_REGION": "us-east-1"},
    )
    if check and r.returncode != 0 and "EntityAlreadyExists" not in r.stderr:
        print(f"  ! aws {' '.join(args[:3])}: {r.stderr.strip()[:200]}")
    return r.returncode, (r.stdout or r.stderr)


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main() -> int:
    _, acct = aws("sts", "get-caller-identity", "--query", "Account", "--output", "text")
    acct = acct.strip()
    provider_arn = f"arn:aws:iam::{acct}:oidc-provider/{PROVIDER_HOST}"

    # 1. the OIDC provider
    rc, _ = aws("iam", "get-open-id-connect-provider",
                "--open-id-connect-provider-arn", provider_arn, check=False)
    if rc != 0:
        aws("iam", "create-open-id-connect-provider",
            "--url", f"https://{PROVIDER_HOST}",
            "--client-id-list", "sts.amazonaws.com",
            # AWS validates GitHub's cert against its own trust store now, but the
            # API still requires this field.
            "--thumbprint-list", "6938fd4d98bab03faadb97b34396831e3780aea1")
        print("  provider   created")
    else:
        print("  provider   already exists")

    # 2. the role, trusting only the two refs
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Federated": provider_arn},
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    f"{PROVIDER_HOST}:aud": "sts.amazonaws.com",
                    # StringEquals, not StringLike, and a list of exact subs.
                    # No wildcard means no fork-PR path in.
                    f"{PROVIDER_HOST}:sub": [
                        f"repo:{REPO}:ref:{ref}" for ref in ALLOWED_REFS
                    ],
                },
            },
        }],
    }
    tp = ROOT / ".oidc-trust.json"
    tp.write_text(json.dumps(trust, indent=1), encoding="utf-8")
    rc, _ = aws("iam", "get-role", "--role-name", ROLE, check=False)
    if rc != 0:
        aws("iam", "create-role", "--role-name", ROLE,
            "--assume-role-policy-document", f"file://{tp}",
            "--description", "GitHub Actions deploy for ragkit (OIDC, no stored keys)")
        print("  role       created")
    else:
        aws("iam", "update-assume-role-policy", "--role-name", ROLE,
            "--policy-document", f"file://{tp}")
        print("  role       trust policy updated")

    # 3. least-privilege deploy policy
    #
    # Lightsail's IAM does not support per-service resource ARNs for these
    # actions, so the scoping that IS available is the action list. Deliberately
    # ABSENT: DeleteContainerService, and anything outside Lightsail. CI can
    # deploy and provision; it cannot tear down.
    perms = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "lightsail:GetContainerServices",
                "lightsail:GetContainerImages",
                "lightsail:GetContainerLog",
                "lightsail:GetContainerServiceDeployments",
                "lightsail:CreateContainerService",
                "lightsail:CreateContainerServiceDeployment",
                "lightsail:CreateContainerServiceRegistryLogin",
                "lightsail:RegisterContainerImage",
                "lightsail:UpdateContainerService",
            ],
            "Resource": "*",
        }],
    }
    pp = ROOT / ".oidc-policy.json"
    pp.write_text(json.dumps(perms, indent=1), encoding="utf-8")
    policy_arn = f"arn:aws:iam::{acct}:policy/{POLICY}"
    rc, _ = aws("iam", "get-policy", "--policy-arn", policy_arn, check=False)
    if rc != 0:
        aws("iam", "create-policy", "--policy-name", POLICY,
            "--policy-document", f"file://{pp}")
        print("  policy     created")
    else:
        aws("iam", "create-policy-version", "--policy-arn", policy_arn,
            "--policy-document", f"file://{pp}", "--set-as-default")
        print("  policy     new version set as default")
    aws("iam", "attach-role-policy", "--role-name", ROLE, "--policy-arn", policy_arn)

    tp.unlink(missing_ok=True)
    pp.unlink(missing_ok=True)

    role_arn = f"arn:aws:iam::{acct}:role/{ROLE}"
    print()
    print("  role arn (not a secret -- it is useless without the OIDC trust):")
    print(f"    {role_arn}")
    print()
    print("  refs allowed to assume it:")
    for ref in ALLOWED_REFS:
        print(f"    repo:{REPO}:ref:{ref}")
    (ROOT / ".oidc-role-arn.txt").write_text(role_arn, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
