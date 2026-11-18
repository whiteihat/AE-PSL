"""Acquire and validate the pinned CMC JPEG mirror without rescaling images."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile

from PIL import Image


REPOSITORY = "asafaa/imagenet100-cmc"
REVISION = "6cc705b814f3a19af8e801d25bb55995ef0ce13e"
CLASSES_SHA256 = "79f3f1552ccc478ce24bd934b4573625db584ddc1f4643ae23c994a94cb11397"
RESOLVE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def metadata(name):
    for attempt in range(12):
        try:
            with urllib.request.urlopen(RESOLVE + name, timeout=60) as response:
                return response.read()
        except OSError as error:
            print(f"Metadata {name}: request {attempt + 1} failed: {error}", flush=True)
            if attempt == 11:
                raise
            time.sleep(min(2 ** min(attempt, 6), 60))


def download(path, expected_size, expected_sha256):
    if path.exists():
        if path.stat().st_size != expected_size or digest(path) != expected_sha256:
            raise ValueError(f"Existing archive differs from pinned mirror: {path}")
        print(f"Verified existing archive {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name}: {expected_size / 1e9:.3f} GB", flush=True)
    for attempt in range(12):
        written = partial.stat().st_size if partial.exists() else 0
        if written == expected_size:
            break
        if written > expected_size:
            raise ValueError(f"Partial file exceeds expected size: {partial}")
        request = urllib.request.Request(RESOLVE + path.name)
        if written:
            request.add_header("Range", f"bytes={written}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if written and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {written}-")
                ):
                    raise ValueError(f"Server did not honor resume request for {path.name}")
                with partial.open("ab" if written else "xb") as stream:
                    last_report = time.monotonic()
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report >= 15:
                            print(f"{path.name}: {written / expected_size:.1%}", flush=True)
                            last_report = time.monotonic()
            if written != expected_size:
                raise OSError(f"Incomplete response: {written}/{expected_size}")
            break
        except OSError as error:
            print(f"{path.name}: download attempt {attempt + 1} failed: {error}", flush=True)
            if attempt == 11:
                raise
            time.sleep(min(2 ** min(attempt, 6), 60))
    if partial.stat().st_size != expected_size or digest(partial) != expected_sha256:
        raise ValueError(f"Downloaded archive failed size or SHA256 check: {partial}")
    partial.rename(path)
    print(f"SHA256 verified: {path.name}", flush=True)


def extract(archive, root, extraction_records):
    marker = extraction_records / (archive.name + ".json")
    if marker.exists():
        print(f"Already extracted: {archive.name}", flush=True)
        return
    count = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] not in {"train", "val"}
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"Unexpected archive entry: {member.filename}")
            target = root / name
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            count += 1
    save_json(marker, {"archive": archive.name, "sha256": digest(archive), "files": count})
    print(f"Extracted {archive.name}: {count} files", flush=True)


def validate(root, provenance, classes, stats):
    fingerprints = provenance / "images.jsonl"
    if fingerprints.exists():
        saved = json.loads((provenance / "validation.json").read_text())
        if (saved.get("repository") != REPOSITORY or saved.get("revision") != REVISION
                or saved.get("class_list_sha256") != CLASSES_SHA256
                or saved.get("image_fingerprints_sha256") != digest(fingerprints)):
            raise ValueError(f"Existing image validation record does not match pinned source: {fingerprints}")
        for split, expected in (("train", 126689), ("val", 5000)):
            observed = Counter()
            with fingerprints.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["split"] == split:
                        observed[row["class"]] += 1
            if sum(observed.values()) != expected or dict(observed) != stats[f"{split}_per_class_counts"]:
                raise ValueError(f"Existing image fingerprints fail the {split} counts check")
        print(json.dumps(saved, indent=2), flush=True)
        return
    counts = {}
    duplicates = []
    train_hashes = set()
    with fingerprints.open("x") as output:
        for split in ("train", "val"):
            split_root = root / split
            actual_classes = sorted(p.name for p in split_root.iterdir() if p.is_dir())
            if actual_classes != sorted(classes):
                raise ValueError(f"Class folders differ from CMC list: {split_root}")
            observed = Counter()
            paths = sorted(p for p in split_root.rglob("*") if p.is_file())
            for index, path in enumerate(paths, 1):
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    if min(width, height) < 1:
                        raise ValueError(f"Empty image: {path}")
                checksum = digest(path)
                if split == "train":
                    train_hashes.add(checksum)
                elif checksum in train_hashes:
                    duplicates.append(path.relative_to(root).as_posix())
                observed[path.parent.name] += 1
                output.write(json.dumps({
                    "path": path.relative_to(root).as_posix(),
                    "split": split,
                    "class": path.parent.name,
                    "sha256": checksum,
                    "width": width,
                    "height": height,
                }) + "\n")
                if index % 10000 == 0:
                    output.flush()
                    print(f"Decoded and hashed {split}: {index}/{len(paths)}", flush=True)
            if dict(observed) != stats[f"{split}_per_class_counts"]:
                raise ValueError(f"Per-class counts differ from pinned mirror: {split}")
            counts[split] = sum(observed.values())
    result = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "class_list_sha256": CLASSES_SHA256,
        "counts": counts,
        "cross_split_duplicate_files": duplicates,
        "image_fingerprints_sha256": digest(fingerprints),
        "image_processing": "JPEG bytes preserved; no resizing or re-encoding",
        "author_class_selection": "Not published in AE-PSL materials",
    }
    save_json(provenance / "validation.json", result)
    if counts != {"train": 126689, "val": 5000}:
        raise ValueError(f"Dataset validation requires review: {result}")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/imagenet100"))
    parser.add_argument("--provenance", type=Path, default=Path("data/provenance/imagenet100-cmc"))
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    root, provenance = args.root.resolve(), args.provenance.resolve()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    classes_path = provenance / "selected_synsets.txt"
    classes_bytes = classes_path.read_bytes() if classes_path.exists() else metadata("selected_synsets.txt")
    classes = classes_bytes.decode().split()
    canonical = ("\n".join(classes) + "\n").encode()
    if len(classes) != 100 or len(set(classes)) != 100 or hashlib.sha256(canonical).hexdigest() != CLASSES_SHA256:
        raise ValueError("Class list differs from the approved CMC list")
    if not classes_path.exists():
        classes_path.write_bytes(classes_bytes)
    labels_path = provenance / "Labels.json"
    if not labels_path.exists():
        labels_path.write_bytes(metadata("Labels.json"))
    stats_path = provenance / "provider-stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else json.loads(metadata("stats.json"))
    save_json(provenance / "provider-stats.json", stats)
    archives_metadata = provenance / "archives.json"
    if archives_metadata.exists():
        entries = json.loads(archives_metadata.read_text())["files"]
    else:
        tree_url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true"
        with urllib.request.urlopen(tree_url, timeout=60) as response:
            tree = json.load(response)
        entries = sorted((entry for entry in tree if entry["path"].endswith(".zip")), key=lambda x: x["path"])
        if len(entries) != 10:
            raise ValueError("Expected ten pinned JPEG archives")
        save_json(archives_metadata, {"repository": REPOSITORY, "revision": REVISION, "files": entries})
    def acquire(entry):
        archive = archives / entry["path"]
        download(archive, entry["size"], entry["lfs"]["oid"])
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        list(pool.map(acquire, entries))
    for entry in entries:
        archive = archives / entry["path"]
        extract(archive, root, provenance / "extracted")
    validate(root, provenance, classes, stats)


if __name__ == "__main__":
    main()
