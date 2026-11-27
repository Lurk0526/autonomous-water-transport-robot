import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, unquote


root = Path(__file__).resolve().parent / "gazebo-ring-v1"
sim_root = Path((root / "mac-gazebo-env.txt").read_text().strip())
packages = json.loads((root / "mac-gazebo-resolved-packages.json").read_text())["actions"]["LINK"]
downloads = sim_root / "verified-downloads"
downloads.mkdir(exist_ok=True)
official = "https://conda.anaconda.org/conda-forge/"
mirror = "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/"


def verified(path, package):
    if not path.is_file() or path.stat().st_size != package["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == package["sha256"]


def fetch(package):
    suffix = f'{package["subdir"]}/{package["fn"]}'
    path = downloads / package["fn"]
    cached = sim_root / "store/pkgs/https/conda.anaconda.org/conda-forge" / suffix
    if verified(path, package):
        return path
    if verified(cached, package):
        return cached
    partial = path.with_suffix(path.suffix + ".partial")
    for origin in (mirror, official, mirror):
        partial.unlink(missing_ok=True)
        result = subprocess.run(
            ["/usr/bin/curl", "--fail", "--silent", "--show-error", "--location",
             "--connect-timeout", "10", "--max-time", "1200", "--speed-limit", "1024",
             "--speed-time", "50", "--output", str(partial), origin + quote(suffix)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and verified(partial, package):
            partial.replace(path)
            return path
        print(f"Retry {suffix}: {result.stderr.strip() or 'size or SHA256 mismatch'}", flush=True)
    raise RuntimeError(f"Could not download and verify {suffix}")


def main():
    for package in packages:
        suffix = f'{package["subdir"]}/{package["fn"]}'
        if (not re.fullmatch(r"(?:noarch|osx-arm64)/[\w.\-!+]+\.(?:conda|tar\.bz2)", suffix)
                or unquote(package["url"]) != official + suffix
                or not re.fullmatch(r"[0-9a-f]{64}", package["sha256"])):
            raise ValueError(f"Unexpected conda-forge package entry: {suffix}")

    resolved = {}
    errors = []
    total = len(packages)
    print(f"Verifying {total} locked conda-forge packages", flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch, package): package["fn"] for package in packages}
        for future in as_completed(futures):
            try:
                resolved[futures[future]] = future.result()
            except RuntimeError as error:
                errors.append(str(error))
                print(error, flush=True)
            checked = len(resolved)
            if checked and (checked % 10 == 0 or checked == total):
                print(f"Verified {checked}/{total} packages", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} dependencies unavailable; Gazebo not installed")

    manifest = sim_root / "gazebo-explicit.txt"
    manifest.write_text("@EXPLICIT\n" + "\n".join(resolved[p["fn"]].as_uri() for p in packages) + "\n")
    subprocess.run(
        [str(sim_root / "bin/micromamba"), "--root-prefix", str(sim_root / "store"), "create",
         "--yes", "--no-rc", "--offline", "--prefix", str(sim_root / "env"), "--file", str(manifest)],
        check=True,
    )


if __name__ == "__main__":
    main()
