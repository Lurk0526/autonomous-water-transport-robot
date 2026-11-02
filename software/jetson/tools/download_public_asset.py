#!/usr/bin/python3
"""Resumable HTTP range download with validated ranges and SHA256 receipt."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('output', type=Path)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--mirror-url', help='Same upstream object through another endpoint; length and ETag must match')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(a.output) + '.part')
    progress = Path(str(a.output) + '.progress.json')
    chunk_size = 16 * 1024 * 1024
    total = (a.size + chunk_size - 1) // chunk_size
    done = set()
    urls = [a.url]
    if a.mirror_url:
        heads = []
        for url in (a.url, a.mirror_url):
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
                heads.append((int(response.headers['Content-Length']), response.headers.get('ETag')))
        if heads[0] != heads[1] or heads[0][0] != a.size or not heads[0][1]:
            raise RuntimeError('Mirror identity did not match source size and ETag')
        urls = [a.mirror_url, a.url]
    if a.output.exists():
        if a.output.stat().st_size != a.size:
            raise RuntimeError('Existing output has unexpected size')
    else:
        if progress.exists() and partial.exists():
            state = json.loads(progress.read_text())
            if state['url'] != a.url or state['size'] != a.size:
                raise RuntimeError('Resume metadata does not match the requested asset')
            done = set(state['completed'])
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, a.size)
        lock = threading.Lock()
        started = time.monotonic()

        def get(index):
            start = index * chunk_size
            end = min(a.size, start + chunk_size) - 1
            for attempt in range(10):
                try:
                    request = urllib.request.Request(urls[attempt % len(urls)], headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        expected = f'bytes {start}-{end}/{a.size}'
                        if response.status != 206 or response.headers.get('Content-Range') != expected:
                            raise RuntimeError(f'Unexpected range response: {response.status}, {response.headers.get("Content-Range")}')
                        offset = start
                        while data := response.read(1024 * 1024):
                            if offset + len(data) > end + 1:
                                raise RuntimeError('Server sent bytes beyond the requested range')
                            view = memoryview(data)
                            while view:
                                wrote = os.pwrite(fd, view, offset)
                                offset += wrote
                                view = view[wrote:]
                        if offset != end + 1:
                            raise RuntimeError('Incomplete range')
                    with lock:
                        done.add(index)
                        state = {'url': a.url, 'size': a.size, 'chunk_size': chunk_size,
                                 'completed': sorted(done), 'total_chunks': total}
                        temp = Path(str(progress) + '.tmp')
                        temp.write_text(json.dumps(state))
                        temp.replace(progress)
                        if len(done) % 5 == 0 or len(done) == total:
                            print(f'{len(done)}/{total} chunks; {len(done)/total:.1%}; elapsed {time.monotonic()-started:.0f}s', flush=True)
                    return
                except Exception:
                    if attempt == 9:
                        raise
                    time.sleep(min(2 ** attempt, 15))

        try:
            with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
                list(pool.map(get, [i for i in range(total) if i not in done]))
            os.fsync(fd)
        finally:
            os.close(fd)
        partial.replace(a.output)
    with a.output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'url': a.url, 'file': str(a.output.resolve()), 'bytes': a.size, 'sha256': digest}
    Path(str(a.output) + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
