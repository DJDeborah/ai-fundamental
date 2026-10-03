"""Run a shell command on a Jupyter server through its terminal WebSocket.

The token is read from JUPYTER_TOKEN and is never written to disk or stdout.
This is an operator helper for an instance that the user already owns.
"""

import argparse
import json
import os
import re
import shlex
import sys
import time
import uuid

import requests
import websocket


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def drain(ws, seconds=0.5):
    deadline = time.monotonic() + seconds
    ws.settimeout(0.1)
    while time.monotonic() < deadline:
        try:
            ws.recv()
        except websocket.WebSocketTimeoutException:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Jupyter base URL ending in /jupyter")
    parser.add_argument("--terminal", default="2")
    parser.add_argument("--cwd", default="/root/ai-fundamental")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    token = os.environ.get("JUPYTER_TOKEN")
    if not token:
        raise SystemExit("Set JUPYTER_TOKEN in the environment")
    base = args.url.rstrip("/")
    if not base.startswith("https://"):
        raise SystemExit("Only HTTPS Jupyter URLs are supported")
    origin = base.split("/jupyter", 1)[0]
    response = requests.get(
        f"{base}/api/terminals",
        headers={"Authorization": f"token {token}"},
        timeout=20,
    )
    response.raise_for_status()
    if args.terminal not in {item["name"] for item in response.json()}:
        raise SystemExit(f"Terminal {args.terminal} does not exist")

    ws_url = f"wss://{base.removeprefix('https://')}/terminals/websocket/{args.terminal}"
    ws = websocket.create_connection(
        ws_url,
        header=[f"Authorization: token {token}"],
        origin=origin,
        timeout=20,
    )
    try:
        drain(ws)
        ws.send(json.dumps(["stdin", "stty -echo\n"]))
        drain(ws)
        marker = f"__CODEX_DONE_{uuid.uuid4().hex}__"
        command = (
            f"( cd {shlex.quote(args.cwd)} && {args.command} ); "
            f"_codex_rc=$?; printf '\\n{marker}:%s\\n' \"$_codex_rc\"\n"
        )
        ws.send(json.dumps(["stdin", command]))
        deadline = time.monotonic() + args.timeout
        output = ""
        while time.monotonic() < deadline:
            ws.settimeout(min(2, max(0.1, deadline - time.monotonic())))
            try:
                message = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if message[0] != "stdout":
                continue
            output += message[1]
            if marker in output:
                before, after = output.split(marker, 1)
                sys.stdout.write(ANSI.sub("", before).replace("\r", ""))
                match = re.match(r":(\d+)", after)
                if not match:
                    raise SystemExit("Remote completion marker was malformed")
                print(f"\nREMOTE_EXIT={match.group(1)}")
                raise SystemExit(int(match.group(1)))
        raise SystemExit(f"Remote command did not finish within {args.timeout:g} seconds")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
