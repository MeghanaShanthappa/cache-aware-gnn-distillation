"""Check restored original results or inputs against the snapshot inventory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',action='store_true');p.add_argument('--inputs',action='store_true')
    args=p.parse_args()
    if not args.results and not args.inputs:p.error('Choose --results and/or --inputs')
    records=[]
    for enabled,name in [(args.results,'result-files.json'),(args.inputs,'input-files.json')]:
        if enabled:records+=json.loads((ROOT/'artifacts'/name).read_text())
    missing=[];mismatched=[]
    for record in records:
        path=ROOT/record['path']
        if not path.exists():missing.append(record['path']);continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
        if path.stat().st_size!=record['bytes'] or h.hexdigest()!=record['sha256']:
            mismatched.append(record['path'])
    print(json.dumps(dict(files=len(records),missing=missing,mismatched=mismatched),indent=2))
    if missing or mismatched:raise SystemExit(1)


if __name__=='__main__':main()
