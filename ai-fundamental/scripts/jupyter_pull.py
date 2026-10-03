"""Download a file from an owned Jupyter server into the local research archive.

Set JUPYTER_TOKEN in the environment. The token is never written to disk.
"""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="HTTPS Jupyter base URL ending in /jupyter")
    parser.add_argument("--remote", required=True, help="absolute path under /root on the server")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--max-gb", type=float, default=2)
    args = parser.parse_args()
    token = os.environ.get("JUPYTER_TOKEN")
    if not token:
        raise SystemExit("Set JUPYTER_TOKEN in the environment")
    if not args.url.startswith("https://") or not args.remote.startswith("/root/"):
        raise ValueError("expected HTTPS Jupyter URL and remote path under /root")
    relative = args.remote.removeprefix("/root/")
    path = quote(relative, safe="/")
    url = f"{args.url.rstrip('/')}/files/{path}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    digest = hashlib.sha256()
    total = 0
    try:
        with requests.get(url, headers={"Authorization": f"token {token}"},
                          stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > args.max_gb * 1e9:
                        raise ValueError("download exceeded --max-gb")
                    digest.update(chunk)
                    handle.write(chunk)
        actual = digest.hexdigest()
        if args.sha256 and actual != args.sha256.lower():
            raise ValueError(f"SHA-256 mismatch: {actual}")
        temporary.replace(args.out)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"{args.out}: {total} bytes SHA-256 {actual}")


if __name__ == "__main__":
    main()
