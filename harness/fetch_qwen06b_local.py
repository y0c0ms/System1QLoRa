"""Parallel resumable fetcher for Qwen/Qwen3-0.6B bytes.

Zscaler blocks the HF CDN (us.aws.cdn.hf.co); cas-bridge via hf_xet works but
hangs without resume. ModelScope hosts the same public model with Range
support, so fetch from there with 8 parallel resumable range workers
(progress tracked in a sidecar JSON), then verify sha256 against the HF
LFS etag.
"""

import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import truststore

REPO = "Qwen/Qwen3-0.6B"
BASE_URL = f"https://modelscope.cn/models/{REPO}/resolve/master"
OUT = Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B"
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "_progress.json"

SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
BIG = ("model.safetensors", 1_503_300_328, SHA)
SMALL = {
    "tokenizer.json": 11_422_654,
    "vocab.json": 2_776_833,
}
N_WORKERS = 8


def client():
    return httpx.Client(timeout=httpx.Timeout(90, read=45), follow_redirects=True,
                        verify=truststore.SSLContext())


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


import threading
_lock = threading.Lock()


def save_state(state: dict):
    with _lock:
        STATE.write_text(json.dumps(state))


def fetch_slice(dest: Path, start: int, end: int, state: dict):
    """Fill bytes [start, end) of dest. ModelScope honors Range but answers
    200 instead of 206, so trust content-range/content-length, not the status
    code. Any other body shape raises and retries."""
    key = dest.name
    with client() as c:
        for attempt in range(300):
            pos = state.get(key, {}).get(str(start), start)
            if pos >= end:
                return f"done [{start}:{end}]"
            try:
                headers = {"Range": f"bytes={pos}-{end - 1}"}
                with c.stream("GET", f"{BASE_URL}/{key}", headers=headers) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    honored = ("content-range" in r.headers
                               or int(r.headers.get("content-length") or -1) == end - pos)
                    if not honored:
                        raise RuntimeError(f"server sent non-range body for [{start}:{end}]")
                    with open(dest, "r+b") as f:
                        f.seek(pos)
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                            pos += len(chunk)
                            state.setdefault(key, {})[str(start)] = pos
                            if pos % (8 << 20) < (1 << 20):
                                save_state(state)
            except Exception as e:
                print(f"  retry {attempt} [{start}:{end}]: {type(e).__name__} {str(e)[:80]}",
                      flush=True)
                save_state(state)
                time.sleep(2)
        save_state(state)
        raise RuntimeError(f"worker [{start}:{end}] exhausted retries")


def fetch_whole(name: str, total: int):
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == total:
        # could still be a hole; the sha check at the end is the authority
        print(f"skip {name} (file present)")
        return
    state = load_state()
    if not dest.exists():
        with open(dest, "wb") as f:
            f.truncate(total)  # sparse pre-allocation; sidecar tracks progress
    bounds = [(i * total // N_WORKERS, (i + 1) * total // N_WORKERS) for i in range(N_WORKERS)]
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(fetch_slice, dest, a, b, state) for a, b in bounds]
        for fut in cf.as_completed(futs):
            print(" ", fut.result(), flush=True)
    if dest.stat().st_size != total:
        sys.exit(f"SIZE MISMATCH {name}: {dest.stat().st_size} != {total}")
    print(f"OK   {name} ({total} bytes)")


def main():
    for name, total in SMALL.items():
        fetch_whole(name, total)
    name, total, sha = BIG
    fetch_whole(name, total)
    h = hashlib.sha256()
    with open(OUT / name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        sys.exit(f"SHA MISMATCH: {h.hexdigest()} != {sha}")
    print("SHA_OK")


if __name__ == "__main__":
    main()
