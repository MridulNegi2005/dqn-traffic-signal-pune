"""Download the Pune dataset from Mendeley Data and check SHA-256 hashes."""
import hashlib
import json
import urllib.request

from config import ROOT

API = "https://data.mendeley.com/public-api/datasets/xnf2k6n288/files?folder_id=root&version=1"
RAW = ROOT / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    files = json.load(urllib.request.urlopen(API, timeout=60))
    for f in files:
        dst = RAW / f["filename"]
        want = f["content_details"]["sha256_hash"]
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == want:
            print("ok  ", dst.name)
            continue
        print("get ", dst.name, f["size"], "bytes")
        urllib.request.urlretrieve(f["content_details"]["download_url"], dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != want:
            raise SystemExit(f"hash mismatch: {dst.name}")


if __name__ == "__main__":
    main()
