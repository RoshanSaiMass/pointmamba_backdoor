#!/usr/bin/env python3
"""Download official ModelNet10 meshes with verified TLS and safe extraction.

Uses Python's standard library; no torch installation is needed. An optional
SHA256 supplied from a trusted source can pin archive identity. ZIP CRC and
expected class/split counts are always verified before publishing extraction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile

URL = 'https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
CLASSES = ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
           'night_stand', 'sofa', 'table', 'toilet')
EXPECTED_COUNTS = {'train': 3991, 'test': 908}
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACTED_BYTES = 5 * 1024**3


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_dataset(root, expected_counts=EXPECTED_COUNTS):
    root = Path(root)
    totals = {'train': 0, 'test': 0}
    for name in CLASSES:
        for split in totals:
            files = list((root / name / split).glob('*.off'))
            if not files:
                raise ValueError(f'Missing meshes: {root / name / split}')
            totals[split] += len(files)
    if expected_counts is not None and totals != expected_counts:
        raise ValueError(f'Unexpected ModelNet10 counts: {totals}; expected {expected_counts}')
    return totals


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != 'https':
            raise ValueError('Refusing a redirect away from HTTPS')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, destination, expected_sha256=None):
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Download URL must use HTTPS')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and sha256(destination) != expected_sha256.lower():
            raise ValueError(f'Cached archive checksum mismatch: {destination}; remove it to download again')
        if not zipfile.is_zipfile(destination):
            raise ValueError(f'Cached archive is not a ZIP: {destination}')
        return destination
    # Unique temporary file prevents concurrent downloads from corrupting each other.
    fd, temporary = tempfile.mkstemp(prefix='modelnet10-', suffix='.part', dir=destination.parent)
    os.close(fd)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'ModelNet10-research-downloader/1.0'})
        opener = urllib.request.build_opener(HTTPSRedirectHandler())
        received = 0
        with opener.open(request, timeout=120) as response, open(temporary, 'wb') as f:
            if urllib.parse.urlparse(response.geturl()).scheme != 'https':
                raise ValueError('Final download URL must use HTTPS')
            length = response.headers.get('Content-Length')
            expected_length = int(length) if length else None
            if expected_length is not None and expected_length > MAX_ARCHIVE_BYTES:
                raise ValueError('Archive exceeds download size limit')
            last_report = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > MAX_ARCHIVE_BYTES:
                    raise ValueError('Archive exceeds download size limit')
                f.write(block)
                if received - last_report >= 32 * 1024**2:
                    print(f'Downloaded {received / 1024**2:.0f} MiB', flush=True)
                    last_report = received
            if expected_length is not None and received != expected_length:
                raise ValueError(f'Incomplete download: {received} bytes, expected {expected_length}')
        if expected_sha256 and sha256(temporary) != expected_sha256.lower():
            raise ValueError('Downloaded archive SHA256 does not match; archive discarded')
        if not zipfile.is_zipfile(temporary):
            raise ValueError('Response was not a ZIP archive; archive discarded')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def extract(archive, output, expected_counts=EXPECTED_COUNTS):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return validate_dataset(output, expected_counts)
    with tempfile.TemporaryDirectory(prefix='.modelnet10-extract-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'ModelNet10'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if sum(m.file_size for m in members) > MAX_EXTRACTED_BYTES:
                raise ValueError('ZIP exceeds extraction size limit')
            destinations = set()
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                    raise ValueError(f'Unsafe ZIP path: {member.filename}')
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f'ZIP symlink rejected: {member.filename}')
                parts = path.parts
                # Only extract the documented class/split/OFF structure.
                if len(parts) != 4 or parts[0] != 'ModelNet10' or parts[1] not in CLASSES or parts[2] not in ('train', 'test') or path.suffix.lower() != '.off':
                    continue
                target = staging.joinpath(*parts[1:])
                if target in destinations:
                    raise ValueError(f'Duplicate ZIP entry: {member.filename}')
                destinations.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Reading through ZipFile checks each extracted member's CRC.
                with z.open(member) as source, target.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
            bad = z.testzip()
            if bad:
                raise ValueError(f'ZIP CRC failure: {bad}')
        counts = validate_dataset(staging, expected_counts)
        os.rename(staging, output)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='datasets/ModelNet10')
    p.add_argument('--archive', default='datasets/ModelNet10.zip')
    p.add_argument('--url', default=URL, help='HTTPS source override for an explicitly trusted mirror')
    p.add_argument('--sha256', help='Optional expected archive SHA256 from a trusted source')
    args = p.parse_args()
    if args.sha256 and (len(args.sha256) != 64 or any(c not in '0123456789abcdefABCDEF' for c in args.sha256)):
        p.error('--sha256 must contain 64 hexadecimal characters')
    output = Path(args.output)
    if output.exists():
        counts = validate_dataset(output)
        print(f'Existing dataset verified: {output.resolve()} {counts}')
        return
    try:
        archive = download(args.url, args.archive, args.sha256)
        counts = extract(archive, output)
        record = dict(source=args.url, archive_sha256=sha256(archive), counts=counts)
        (output / 'download_manifest.json').write_text(json.dumps(record, indent=2))
    except Exception as e:
        p.exit(1, f'ModelNet10 download failed: {e}\nExisting files were preserved. Check source availability/network access; a trusted HTTPS mirror can be supplied with --url.\n')
    print(f'ModelNet10 ready: {output.resolve()} {counts}', flush=True)


if __name__ == '__main__':
    main()
