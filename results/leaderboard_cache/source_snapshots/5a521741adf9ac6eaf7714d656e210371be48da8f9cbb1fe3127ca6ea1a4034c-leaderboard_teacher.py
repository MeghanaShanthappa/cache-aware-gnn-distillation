"""Train released top-ten architectures with held-out cache calibration labels.

AGDN uses authors' model and train/evaluate functions. SAGN uses authors'
model, supervised/consistency updates and EMA. Only data plumbing and checkpoint
selection/storage are supplied here; upstream source trees are unchanged.
"""
import argparse
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import types

import leaderboard_common as C
from leaderboard_common import torch, np, F, ROOT, OUT, DATA


def log(**items):
    print(json.dumps(items), flush=True)


def arxiv(args, target):
    import dgl
    import pandas as pd
    srcpath = ROOT / 'leaderboard-upstream/AGDN/ogbn-arxiv/src'
    sys.path.insert(0, str(srcpath))
    source = importlib.import_module('main')
    config = types.SimpleNamespace(use_labels=True, mask_rate=.5, n_label_iters=1,
        use_consis_loss=False, mode='teacher', model='agdn', no_attn_dst=False,
        no_residual=False, no_bias_last=False, no_bias=False, n_hidden=256,
        n_layers=2, n_heads=3, dropout=.85, input_drop=.35, edge_drop=.6,
        attn_drop=0., diffusion_drop=.3, K=3, no_position_emb=False,
        transition_matrix='gat_sym', weight_style='HA', HA_activation='leakyrelu',
        zero_inits=False, alpha=.95, temp=.75)
    split = C.load_split(args.dataset)
    labels = torch.as_tensor(np.load(ROOT/'ogb-data/ogbn-arxiv/labels.npy'), device='cuda')[:,None]
    x = torch.as_tensor(np.load(DATA/'arxiv-giant.npy'), device='cuda')
    edge = pd.read_csv(ROOT/'ogb-data/arxiv/raw/edge.csv.gz', header=None).values
    graph = dgl.graph((torch.as_tensor(edge[:,0]), torch.as_tensor(edge[:,1])), num_nodes=len(x))
    src, dst = graph.edges()
    graph.add_edges(dst, src)
    graph = graph.remove_self_loop().add_self_loop().to('cuda')
    graph.ndata['feat'] = x
    inv, sqrt, isqrt = source.compute_norm(graph)
    graph.srcdata['src_norm'] = isqrt
    graph.dstdata['dst_norm'] = isqrt
    graph.apply_edges(dgl.function.u_mul_v('src_norm','dst_norm','gcn_norm'))
    graph.dstdata['dst_norm'] = sqrt
    graph.apply_edges(dgl.function.u_mul_v('src_norm','dst_norm','gcn_norm_adjust'))
    graph.edata['sage_norm'] = inv[graph.edges()[1]]
    source.device, source.in_feats, source.n_classes = torch.device('cuda'), x.shape[1], labels.max().item()+1
    train = torch.as_tensor(split['train'], device='cuda')
    valid = torch.as_tensor(split['model_valid'], device='cuda')
    # The evaluator function uses this vector only for label reuse and logging.
    # Pass every remaining unlabelled node, so calibration nodes receive exactly
    # the same iterative label-feature treatment without revealing their labels.
    rest = torch.as_tensor(np.concatenate([split['calibration'],split['test']]), device='cuda')
    from ogb.nodeproppred import Evaluator
    evaluator = Evaluator('ogbn-arxiv')
    C.write(target/'protocol.json', C.metadata(args.dataset,args.seed,dict(vars(config),
        epochs_per_stage=args.arxiv_epochs,optimizer='RMSprop',learning_rate=.01,
        weight_decay=0,selection='minimum model-validation loge cross entropy',
        deviations=['Half official validation reserved exclusively for cache calibration',
                    'Native CUDA DGL 1.1.2; authors model and train/evaluate functions unchanged',
                    'Best state and logits saved, test quality computed only after selection'])))
    teacher_output = None
    for stage in ['supervised','self_kd']:
        checkpoint = target/(stage+'.pt')
        if checkpoint.exists() and (target/(stage+'-complete.json')).exists():
            teacher_output = torch.load(checkpoint,map_location='cuda')['logits']
            continue
        C.seed_all(args.seed)
        dgl.seed(args.seed)
        config.mode = 'teacher' if stage=='supervised' else 'student'
        model = source.gen_model(x.shape[1], source.n_classes, config).cuda()
        opt = torch.optim.RMSprop(model.parameters(),lr=.01,weight_decay=0)
        best, begin = float('inf'), time.perf_counter()
        trace = []
        for epoch in range(1,args.arxiv_epochs+1):
            acc, loss = source.train(config,model,graph,labels,train,valid,rest,opt,
                teacher_output,source.loge_cross_entropy,evaluator,epoch)
            with torch.no_grad():
                values = source.evaluate(config,model,graph,labels,train,valid,rest,
                    True,source.loge_cross_entropy,evaluator)
                val_loss = values[4].item()
                if val_loss < best:
                    best = val_loss
                    torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},
                        logits=values[-1].cpu(),epoch=epoch,validation_accuracy=values[1],
                        validation_loss=val_loss,parameters=sum(p.numel() for p in model.parameters())), checkpoint)
            if epoch%20==0 or epoch==1:
                row = dict(stage=stage,epoch=epoch,loss=loss.item(),validation_accuracy=values[1],
                    validation_loss=val_loss,best_validation_loss=best,seconds=time.perf_counter()-begin)
                trace.append(row); C.write(target/(stage+'-trace.json'),trace)
                log(dataset=args.dataset,seed=args.seed,**row)
        teacher_output = torch.load(checkpoint,map_location='cuda')['logits']
        C.write(target/(stage+'-complete.json'),dict(epochs=args.arxiv_epochs,
            checkpoint_sha256=C.digest(checkpoint)))
        del model,opt,values
        torch.cuda.empty_cache()
    torch.save(torch.load(target/'self_kd.pt',map_location='cpu'),target/'teacher.pt')


def products_preprocess():
    import scipy.sparse as sp
    directory = DATA/'products-preprocessed'
    directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists():
        if json.loads((directory/'complete.json').read_text()).get('graph_protocol')=='raw-original-loops-v2':
            return directory
    a = sp.load_npz(ROOT/'ogb-data/ogbn-products/adjacency.npz').tocsr()
    # Existing file has symmetric-normalized weights and inserted loops. Authors'
    # SAGN preprocessing uses the raw undirected graph and its row mean.
    a.data.fill(1)
    a.setdiag(0); a.eliminate_zeros();a.sort_indices()
    # OGB products contains 128 original self-edges, each duplicated by the
    # official inverse-edge loader. Restore these, not the artificially inserted
    # loop at every node. Non-loop edges have no duplicates (verified raw audit).
    audit_path=OUT/'raw-products-edge-audit.json'
    audit=json.loads(audit_path.read_text())
    assert a.nnz==2*(audit['raw_edges']-audit['self_loops'])
    diagonal=np.zeros(a.shape[0],dtype=np.float32)
    np.add.at(diagonal,np.asarray(audit['loop_ids'],dtype=np.int64),2)
    a.setdiag(diagonal);a.eliminate_zeros();a.sort_indices()
    degree = np.asarray(a.sum(1)).ravel().astype(np.float32)
    a.data *= np.repeat(1/np.maximum(degree,1),np.diff(a.indptr))
    matrix = torch.sparse_csr_tensor(torch.as_tensor(a.indptr.astype(np.int64),device='cuda'),
        torch.as_tensor(a.indices.astype(np.int64),device='cuda'),
        torch.as_tensor(a.data,device='cuda'),size=a.shape)
    x = torch.as_tensor(np.load(DATA/'products-giant.npy'),device='cuda')
    for hop in range(4):
        np.save(directory/f'features-{hop}.npy',x.cpu().numpy())
        if hop<3:x=torch.sparse.mm(matrix,x)
        log(phase='products_preprocessing',feature_hop=hop)
    labels = np.load(ROOT/'ogb-data/ogbn-products/labels.npy')
    train = C.load_split('ogbn-products')['train']
    y = torch.zeros((len(x),int(labels.max()+1)),device='cuda')
    y[torch.as_tensor(train,device='cuda'),torch.as_tensor(labels[train],device='cuda')] = 1
    for hop in range(9):
        y = torch.sparse.mm(matrix,y)
    np.save(directory/'label-embedding.npy',y.cpu().numpy())
    C.write(directory/'complete.json',dict(feature_hops=3,label_hops=9,
        row_normalized=True,self_loops='128 original edges, doubled as in OGB loader',edges=a.nnz,
        graph_protocol='raw-original-loops-v2',raw_directed_edge_entries=2*audit['raw_edges'],
        labels_used='official training labels only',
        giant_sha256=C.digest(DATA/'products-giant.npy'),
        label_embedding_sha256=C.digest(directory/'label-embedding.npy')))
    return directory


def products(args,target):
    directory = products_preprocess()
    torch.cuda.empty_cache()
    srcpath = ROOT/'leaderboard-upstream/SCR/ogbn-products'
    sys.path.insert(0,str(srcpath))
    # utils imports the CogDL loader at module scope; training/model functions
    # do not use it. Supply an explicit unused stub, avoiding a second graph copy.
    loader_stub = types.ModuleType('load_dataset')
    def unused(*a,**k):raise RuntimeError('Upstream CogDL loader is not used')
    loader_stub.load_dataset = unused
    sys.modules['load_dataset'] = loader_stub
    source = importlib.import_module('utils')
    config = types.SimpleNamespace(method='SAGN',num_hops=3,hidden=256,mlp_layer=1,
        num_heads=1,weight_style='attention',dropout=.5,input_drop=.2,att_drop=.4,
        zero_inits=True,position_emb=False,focal='first',label_mlp_layer=4,
        label_drop=.5,label_residual=False,tem=.5,lam=.5,kl_lam=.2,kl=True,ema_decay=0.)
    split = C.load_split(args.dataset)
    labels = torch.as_tensor(np.load(ROOT/'ogb-data/ogbn-products/labels.npy'))
    # Retain features on GPU. Upstream mini-batches index the same unchanged rows.
    features = [torch.as_tensor(np.load(directory/f'features-{i}.npy'),device='cuda') for i in range(4)]
    label_emb = torch.as_tensor(np.load(directory/'label-embedding.npy'),device='cuda')
    classes = int(labels.max()+1)
    C.seed_all(args.seed)
    model = source.gen_model_sagn(config,features[0].shape[1],classes,classes).cuda()
    mean_teacher = source.gen_model_sagn(config,features[0].shape[1],classes,classes).cuda()
    for p in mean_teacher.parameters():p.detach_()
    ema = source.EMA(model,.9);ema.register()
    opt = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    loader = lambda ix,bs,shuffle=False:torch.utils.data.DataLoader(ix,batch_size=bs,shuffle=shuffle,drop_last=False)
    train_loader = loader(torch.as_tensor(split['train']),50000,True)
    all_loader = loader(torch.arange(len(labels)),50000)
    criterion = torch.nn.CrossEntropyLoss()
    C.write(target/'protocol.json', C.metadata(args.dataset,args.seed,dict(vars(config),
        epochs=args.products_epochs,warm_up=100,confidence_top=.85,confidence_down=.8,
        gap=20,batch_size=50000,optimizer='Adam',learning_rate=.001,
        ema_decay=.9,selection='maximum model-validation accuracy on EMA weights',
        deviations=['Half official validation reserved exclusively for cache calibration',
            'GPU sparse row-mean preprocessing replaces CogDL CPU sparse multiplication',
            'Best EMA checkpoint saved explicitly; avoids released final-EMA checkpoint mixup',
            'Validation only every 10 epochs; no test accuracy used during selection'])))
    best,begin,trace = -1,time.perf_counter(),[]
    for epoch in range(1,args.products_epochs+1):
        if epoch<=100:
            source.train_sagn('cuda',model,features,label_emb,labels,criterion,opt,train_loader,config,ema)
        else:
            if (epoch-1)%20==0 or epoch==101:
                with torch.no_grad():
                    prob = source.gen_output_torch(model,features,all_loader,'cuda',label_emb,config).softmax(1)
                threshold = .85-(.85-.8)*epoch/args.products_epochs
                eligible = torch.as_tensor(np.concatenate([split['valid'],split['test']]))
                confident = eligible[prob[eligible].max(1).values>threshold]
                if not len(confident):raise RuntimeError('No confident pseudo nodes')
                batch_u = int(50000*len(confident)/(len(confident)+len(split['train'])))
                batch_t = int(50000*len(split['train'])/(len(confident)+len(split['train'])))
                enhance = loader(confident,batch_u,True)
                mixed_train = loader(torch.as_tensor(split['train']),batch_t,True)
                del prob
                log(dataset=args.dataset,seed=args.seed,epoch=epoch,confident_nodes=len(confident))
            source.train_mean_teacher_sagn('cuda',model,mean_teacher,features,label_emb,labels,
                criterion,opt,mixed_train,enhance,config,ema)
        if epoch%10==0 or epoch==1:
            ema.apply_shadow();model.eval()
            with torch.no_grad():
                valid = torch.as_tensor(split['model_valid'],device='cuda')
                z,_ = model([x[valid] for x in features],label_emb[valid])
                accuracy = (z.argmax(1).cpu()==labels[split['model_valid']]).float().mean().item()
                if accuracy>best:
                    best = accuracy
                    torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},
                        epoch=epoch,validation_accuracy=best,
                        parameters=sum(p.numel() for p in model.parameters())),target/'best-state.pt')
            ema.restore()
            row=dict(epoch=epoch,validation_accuracy=accuracy,best_validation_accuracy=best,
                seconds=time.perf_counter()-begin)
            trace.append(row);C.write(target/'trace.json',trace)
            log(dataset=args.dataset,seed=args.seed,**row)
    best_state=torch.load(target/'best-state.pt',map_location='cuda')
    model.load_state_dict(best_state['state'])
    with torch.no_grad():
        z=source.gen_output_torch(model,features,all_loader,'cuda',label_emb,config)
    best_state['logits']=z
    best_state['state']={k:v.cpu() for k,v in best_state['state'].items()}
    torch.save(best_state,target/'teacher.pt')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['ogbn-arxiv','ogbn-products'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--arxiv-epochs',type=int,default=2000)
    p.add_argument('--products-epochs',type=int,default=400)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();args.output=args.output.resolve();C.setup()
    target=args.output/args.dataset/f'seed-{args.seed}'
    target.mkdir(parents=True,exist_ok=True)
    if (target/'teacher-complete.json').exists():return
    if args.dataset=='ogbn-arxiv':arxiv(args,target)
    else:products(args,target)
    checkpoint=torch.load(target/'teacher.pt',map_location='cpu')
    split=C.load_split(args.dataset)
    labels=torch.as_tensor(np.load(ROOT/'ogb-data'/args.dataset/'labels.npy'))
    pred=checkpoint['logits'].argmax(1)
    score={k:(pred[v]==labels[v]).float().mean().item() for k,v in split.items()}
    C.write(target/'teacher-complete.json',dict(accuracy=score,epoch=checkpoint['epoch'],
        teacher_sha256=C.digest(target/'teacher.pt'),parameters=checkpoint['parameters']))
    log(dataset=args.dataset,seed=args.seed,phase='teacher_complete',**score)


if __name__=='__main__':main()
