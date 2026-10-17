"""Download the published codec weights and check each file's size and sha256 against weights/manifest.json.

    python scripts/download_weights.py [--to DIR] [--only NAME ...]
    python scripts/download_weights.py --coco-splits COCO_ROOT   (the COCO split annotation files instead,
                                                                  into COCO_ROOT/annotations/)

Each file comes from the Arbutus mirror first and from Zenodo if the mirror fails or serves a file that does not
match the manifest. A file already in DIR that matches is not downloaded again. A file is written under its final
name only after it has passed the check. With --coco-splits, the split archive is downloaded and checked the same way,
then each file in it is checked against the manifest as it is unpacked, and the archive is removed. Exits 1 if any
file could not be fetched intact. Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARBUTUS_BASE = 'https://object-arbutus.alliancecan.ca/6c162a6eb85b4659a5210b4712832d98:'
MIRROR = ARBUTUS_BASE + 'cfm-public/cfwm/v1.0/{name}'
ZENODO = 'https://zenodo.org/records/23408855/files/{name}?download=1'   # DOI 10.5281/zenodo.23408855
PLACEHOLDER = 'TODO-HOSTING'
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def intact(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def fetch(url: str, dst: Path, entry: dict):
    """Download url to dst if it matches the manifest entry. Returns None on success, else the reason."""
    partial = dst.with_name(dst.name + '.partial')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'cfwm-download-weights'})
        h, n = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, 'wb') as f:
            for block in iter(lambda: response.read(CHUNK), b''):
                f.write(block)
                h.update(block)
                n += len(block)
        if n != entry['size']:
            return f'{n} bytes, the manifest says {entry["size"]}'
        if h.hexdigest() != entry['sha256']:
            return 'sha256 differs from the manifest'
        os.replace(partial, dst)
        return None
    except (urllib.error.URLError, OSError) as e:
        return str(e)
    finally:
        partial.unlink(missing_ok=True)


def unpack(archive: Path, dest: Path, members: list) -> str | None:
    """Unpack the manifest's members of archive into dest, each checked. Returns None on success, else the reason."""
    want = {m['name']: m for m in members}
    written = []
    try:
        with tarfile.open(archive) as tar:
            for info in tar:
                m = want.pop(info.name, None)
                if m is None or not info.isfile():
                    return f'unexpected entry {info.name!r} in the archive'
                dst = dest / info.name
                partial = dst.with_name(dst.name + '.partial')
                h, n = hashlib.sha256(), 0
                with tar.extractfile(info) as src, open(partial, 'wb') as f:
                    for block in iter(lambda: src.read(CHUNK), b''):
                        f.write(block)
                        h.update(block)
                        n += len(block)
                if n != m['size'] or h.hexdigest() != m['sha256']:
                    partial.unlink()
                    return f'{info.name} differs from the manifest'
                written.append((partial, dst))
        if want:
            return f'not in the archive: {", ".join(want)}'
        for partial, dst in written:
            os.replace(partial, dst)
        written = []
        return None
    except (tarfile.TarError, OSError) as e:
        return str(e)
    finally:
        for partial, _ in written:
            partial.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--to', type=Path, default=ROOT / 'weights',
                    help='Directory to save the weights in (default: weights/)')
    ap.add_argument('--only', nargs='+', metavar='NAME',
                    help='Only these files, e.g. FasterRCNN or FasterRCNN.ckpt (default: every file in the manifest)')
    ap.add_argument('--coco-splits', type=Path, metavar='COCO_ROOT',
                    help='Instead of the weights: the COCO split annotation files, unpacked into '
                         'COCO_ROOT/annotations/ (COCO_ROOT is the detection --data_root)')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'weights' / 'manifest.json', help=argparse.SUPPRESS)
    ap.add_argument('--mirror', default=MIRROR, help='URL template of the mirror, with {name} (default: Arbutus)')
    ap.add_argument('--zenodo', default=ZENODO, help='URL template of the fallback, with {name} (default: Zenodo)')
    args = ap.parse_args(argv)

    if args.coco_splits and args.only:
        ap.error('--coco-splits and --only do not go together')
    manifest = json.loads(args.manifest.read_text())
    entries = manifest['splits'] if args.coco_splits else manifest['files']
    dest = args.coco_splits / 'annotations' if args.coco_splits else args.to
    if args.only:
        by_name = {e['name']: e for e in entries} | {Path(e['name']).stem: e for e in entries}
        unknown = [n for n in args.only if n not in by_name]
        if unknown:
            ap.error(f'not in the manifest: {", ".join(unknown)}')
        entries = list({by_name[n]['name']: by_name[n] for n in args.only}.values())
    hosts = [(label, url) for label, url in (('mirror', args.mirror), ('Zenodo', args.zenodo))
             if PLACEHOLDER not in url]
    if not hosts:
        print('The download locations of the weights are not published yet.', file=sys.stderr)
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for entry in entries:
        dst = dest / entry['name']
        if all(intact(dest / m['name'], m) for m in entry['members']) if args.coco_splits else intact(dst, entry):
            print(f'{entry["name"]}: already present')
            continue
        for label, url in hosts:
            reason = fetch(url.format(name=entry['name']), dst, entry)
            if reason is None and args.coco_splits:
                reason = unpack(dst, dest, entry['members'])
                dst.unlink()
            if reason is None:
                print(f'{entry["name"]}: from {label}, {entry["size"] / 1e6:.1f} MB, sha256 checked')
                break
            print(f'{entry["name"]}: {label} failed ({reason})')
        else:
            failed.append(entry['name'])
    if failed:
        print(f'Not downloaded: {", ".join(failed)}', file=sys.stderr)
        return 1
    count = sum(len(e['members']) for e in entries) if args.coco_splits else len(entries)
    print(f'{count} file(s) in {dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
