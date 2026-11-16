"""Verify and restore selected groups of split release archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--groups',nargs='+');p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();manifest=json.loads((ROOT/'artifacts/release-manifest.json').read_text())
    groups=manifest['groups']
    if args.groups:
        unknown=set(args.groups)-{g['name'] for g in groups}
        if unknown:raise SystemExit('Unknown archive groups: '+', '.join(sorted(unknown)))
        groups=[g for g in groups if g['name'] in args.groups]
    for group in groups:
        parts=sorted(group['parts'],key=lambda x:x['part'])
        for part in parts:
            path=args.assets/part['name']
            if path.stat().st_size!=part['bytes'] or sha256(path)!=part['sha256']:
                raise RuntimeError('Archive checksum mismatch: '+str(path))
        print(group['name']+': all part checksums verified',flush=True)
        if args.verify_only:continue
        zstd=subprocess.Popen(['zstd','-d','-q','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        # Feed parts from a thread while tarfile consumes the decompressed stream.
        import threading
        failures=[]
        def feed():
            try:
                for part in parts:
                    with (args.assets/part['name']).open('rb') as stream:
                        shutil.copyfileobj(stream,zstd.stdin,8*1024*1024)
            except Exception as error:failures.append(error)
            finally:zstd.stdin.close()
        worker=threading.Thread(target=feed);worker.start()
        try:
            with tarfile.open(fileobj=zstd.stdout,mode='r|') as archive:
                for member in archive:
                    target=(ROOT/member.name).resolve()
                    if target!=ROOT and ROOT not in target.parents:
                        raise RuntimeError('Unsafe archive member: '+member.name)
                    if not (member.isfile() or member.isdir()):
                        raise RuntimeError('Unsupported archive member type: '+member.name)
                    # Extraction is limited to recorded project-relative paths.
                    if not any(member.name==prefix or member.name.startswith(prefix.rstrip('/')+'/')
                               for prefix in group['members']):
                        raise RuntimeError('Unexpected archive prefix: '+member.name)
                    archive.extract(member,ROOT)
        except BaseException:
            zstd.terminate();raise
        finally:
            zstd.stdout.close();worker.join();code=zstd.wait()
        if code or failures:raise RuntimeError('Archive extraction failed: '+group['name'])
        print(group['name']+': restored',flush=True)


if __name__=='__main__':main()
