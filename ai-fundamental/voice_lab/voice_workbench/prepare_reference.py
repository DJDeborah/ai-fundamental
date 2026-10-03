"""Download one fixed public-domain LJ Speech reference, never a celebrity sample."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
EXPECTED = "d4c1fdbe920c2efeb525dcc067ed01c7feaa07ded719a98bd7e62a8f277cd0cd"


def main():
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise RuntimeError("The public sample differs from the verified reference")
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "ljspeech_reference.wav").write_bytes(data)
    receipt = {"dataset": "LJ Speech", "speaker": "Linda Johnson",
               "identity_is_adele": False, "license": "Public Domain",
               "source_url": URL, "license_source": "https://keithito.com/LJ-Speech-Dataset/",
               "sha256": EXPECTED, "bytes": len(data),
               "role": "speech baseline reference, not Adele or a singing dataset"}
    (assets / "LJ_REFERENCE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print("LJ Speech reference verified. Speaker: Linda Johnson, not Adele.")


if __name__ == "__main__":
    main()
