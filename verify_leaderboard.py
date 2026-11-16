"""Verify completed student predictions against restored weights on the GPU."""
import json
from pathlib import Path
import leaderboard_common as C
from leaderboard_common import torch,np,ROOT,OUT,DATA


def verified_source(filename,expected):
    actual=ROOT/filename
    if actual.exists() and C.digest(actual)==expected:return str(actual)
    snapshot=OUT/'source_snapshots'/(expected+'-'+filename)
    assert snapshot.exists() and C.digest(snapshot)==expected,(filename,expected)
    return str(snapshot)


def main():
    C.setup();results=[]
    for complete in sorted(OUT.glob('ogbn-*/seed-*/complete-giant.json')):
        folder=complete.parent;name=folder.parent.name
        info=json.loads(complete.read_text());protocol=json.loads((folder/'protocol.json').read_text())
        assert C.digest(folder/'student-giant.pt')==info['student_checkpoint_sha256']
        teacher=torch.load(folder/'teacher.pt',map_location='cpu')
        student=torch.load(folder/'student-giant.pt',map_location='cpu')
        assert C.digest(folder/'teacher.pt')==student['teacher_sha256']
        saved=np.load(folder/'predictions-giant.npz')
        np.testing.assert_array_equal(teacher['logits'].argmax(1).numpy(),saved['teacher'])
        classes=teacher['logits'].shape[1]
        x=torch.as_tensor(np.load(DATA/(name[5:]+'-giant.npy')),device='cuda')
        model=C.Student(x.shape[1],classes).cuda()
        model.load_state_dict(student['state'])
        z=C.infer(model,x)
        np.testing.assert_array_equal(z.argmax(1).cpu().numpy(),saved['student'])
        sources={key:verified_source(key,expected) for key,expected in [
            ('leaderboard_teacher.py',protocol['launcher_sha256']),
            ('leaderboard_common.py',protocol['common_sha256']),
            ('leaderboard_distill.py',info['source_sha256']),
            ('leaderboard_common.py',info['common_sha256'])]}
        for relative,expected in protocol['source_sha256'].items():
            assert C.digest(ROOT/relative)==expected,relative
        assert C.digest(DATA/(name[5:]+'-giant.npy'))==protocol['feature_sha256']
        results.append(dict(dataset=name,seed=int(folder.name[5:]),
            restored_student_weights_match_all_saved_predictions=True,
            teacher_checkpoint_logits_match_all_saved_predictions=True,
            teacher_and_student_checkpoint_hashes_match=True,
            upstream_sources_unchanged=True,feature_hash_matches=True,
            executed_source_files_or_archives_verified=sources))
        print(json.dumps(results[-1]),flush=True)
        del x,model,z,teacher,student
        torch.cuda.empty_cache()
    (OUT/'checkpoint_verification.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
