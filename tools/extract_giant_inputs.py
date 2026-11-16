"""Extract only the published node embeddings from preserved GIANT archives."""
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
for short in ['arxiv','products']:
    source=ROOT/'leaderboard-data'/('ogbn-'+short+'.tar.gz')
    target=ROOT/'leaderboard-data'/(short+'-giant.npy')
    if target.exists():raise SystemExit('Refusing to overwrite existing '+str(target))
    with tarfile.open(source) as archive:
        matches=[m for m in archive if m.isfile() and m.name.endswith('/X.all.xrt-emb.npy')]
        if len(matches)!=1:raise RuntimeError('Expected one embedding array')
        with archive.extractfile(matches[0]) as original,target.open('wb') as destination:
            shutil.copyfileobj(original,destination,8*1024*1024)
    print(target)
