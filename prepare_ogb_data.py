"""Prepare official OGB CSV datasets without loading pickled graph objects."""
import argparse
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from ogb.io.read_graph_raw import read_csv_graph_raw


def download(url, temporary):
    # Keep a valid prefix and fetch bounded ranges concurrently when supported.
    request=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(request,timeout=120) as response:
        if response.status!=206:
            with temporary.open('wb') as stream:
                shutil.copyfileobj(response,stream,length=1024*1024)
            return
        total=int(response.headers['Content-Range'].split('/')[-1])
    offset=temporary.stat().st_size if temporary.exists() else 0
    if offset>total:
        raise RuntimeError('Partial download exceeds the remote file size')
    ranges=[(start,min(start+16*1024*1024,total)-1) for start in range(offset,total,16*1024*1024)]
    def fetch(bounds):
        start,end=bounds
        part=temporary.with_name(temporary.name+f'.range-{start}-{end}')
        if part.exists() and part.stat().st_size==end-start+1:
            return part
        pending=part.with_suffix(part.suffix+'.pending')
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=120) as response:
            expected=f'bytes {start}-{end}/{total}'
            if response.status!=206 or response.headers.get('Content-Range')!=expected:
                raise RuntimeError('Server returned an unexpected byte range')
            with pending.open('wb') as stream:
                shutil.copyfileobj(response,stream,length=1024*1024)
        if pending.stat().st_size!=end-start+1:
            raise RuntimeError('Incomplete byte range')
        pending.replace(part)
        return part
    print(f'Resuming at {offset:,} of {total:,} bytes with four connections',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(fetch,ranges))
    with temporary.open('ab') as destination:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source,destination,length=1024*1024)
            part.unlink()
    if temporary.stat().st_size!=total:
        raise RuntimeError('Download size mismatch')


def prepare(name, root):
    short = name.removeprefix('ogbn-')
    destination = Path(root)/name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'metadata.json').exists():
        return destination
    archive = Path(root)/(short+'.zip')
    raw = Path(root)/short/'raw'
    if not (Path(root)/short/'.extracted').exists():
        if not archive.exists():
            url = f'https://snap.stanford.edu/ogb/data/nodeproppred/{short}.zip'
            print(f'Downloading {url}', flush=True)
            temporary = archive.with_suffix('.zip.partial')
            download(url,temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root)
        (Path(root)/short/'.extracted').touch()
    print(f'Reading {name}', flush=True)
    graph = read_csv_graph_raw(str(raw), add_inverse_edge=(short=='products'))[0]
    n = graph['num_nodes']
    np.save(destination/'features.npy', graph['node_feat'].astype(np.float32))
    labels = pd.read_csv(raw/'node-label.csv.gz', header=None).values.reshape(-1).astype(np.int64)
    np.save(destination/'labels.npy', labels)
    edge = graph['edge_index']
    adjacency = sp.csr_matrix((np.ones(edge.shape[1], dtype=np.float32), (edge[1],edge[0])), shape=(n,n))
    if short == 'arxiv':
        adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(1)
    adjacency.sum_duplicates()
    adjacency.data.fill(1)
    adjacency.sort_indices()
    degrees = np.asarray(adjacency.sum(1)).reshape(-1)
    scale = sp.diags(degrees ** -.5)
    adjacency = (scale @ adjacency @ scale).tocsr().astype(np.float32)
    adjacency.sort_indices()
    sp.save_npz(destination/'adjacency.npz', adjacency)
    split_name = 'time' if short == 'arxiv' else 'sales_ranking'
    splits = {part: pd.read_csv(Path(root)/short/'split'/split_name/(part+'.csv.gz'), header=None).values.reshape(-1).astype(np.int64)
              for part in ['train','valid','test']}
    np.savez_compressed(destination/'splits.npz', **splits)
    info = dict(dataset=name, nodes=n, normalized_adjacency_entries=adjacency.nnz,
                features=graph['node_feat'].shape[1], classes=int(labels.max())+1,
                split=split_name, split_counts={k:len(v) for k,v in splits.items()},
                preprocessing='OGB original features; undirected binary graph; self loops; symmetric degree normalization',
                source=f'https://ogb.stanford.edu/docs/nodeprop/#{name}')
    (destination/'metadata.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info), flush=True)
    return destination


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['ogbn-arxiv','ogbn-products'])
    parser.add_argument('--root',default='/tmp/cache-gnn-ogb')
    args=parser.parse_args()
    for name in args.datasets:
        if name not in ['ogbn-arxiv','ogbn-products']:
            parser.error('Supported datasets: ogbn-arxiv, ogbn-products')
        prepare(name,args.root)
