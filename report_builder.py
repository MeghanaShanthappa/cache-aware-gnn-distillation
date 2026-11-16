"""Render the current report with one consolidated important-results section."""
import json
import hashlib
import re
import statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def leaderboard_addition():
    directory = ROOT / 'results/leaderboard_cache'
    rows = json.loads((directory / 'metrics.json').read_text())
    verified = json.loads((directory / 'independent_verification.json').read_text())
    checkpoints = json.loads((directory / 'checkpoint_verification.json').read_text())
    assert len(rows) == 26 and verified['quality_and_hit_values_verified'] == 130
    assert len(checkpoints) == 2 and all(r['restored_student_weights_match_all_saved_predictions'] for r in checkpoints)
    data = {}
    table_rows = []
    for name in ['ogbn-arxiv', 'ogbn-products']:
        selected = [r for r in rows if r['dataset'] == name and r['seed'] == 42]
        assert len(selected) == 13
        by_policy = {p: next(r for r in selected if r['policy'] == p and r['budget'] == (0 if p == 'none' else .1))
                     for p in ['none', 'frequency', 'disagreement', 'benefit']}
        complete = json.loads((directory / name / 'seed-42/complete-giant.json').read_text())
        nodes = json.loads((ROOT / 'ogb-data' / name / 'metadata.json').read_text())['nodes']
        data[name] = dict(policies=by_policy, teacher=complete['teacher_accuracy'] * 100, nodes=nodes)
        table_rows.append('| ' + ' | '.join([name, 'GIANT A100', f"{complete['teacher_accuracy'] * 100:.2f}"] + [f"{by_policy[p]['accuracy'] * 100:.2f}" for p in by_policy]) + ' |')
    arxiv, products = data['ogbn-arxiv'], data['ogbn-products']
    a, p = arxiv['policies']['benefit'], products['policies']['benefit']
    return dict(
        rows=table_rows,
        summary=("**Teacher-level accuracy in the GIANT study.** Adding a limited cache of teacher predictions recovered nearly the entire teacher–student accuracy gap on both datasets, achieving approximately teacher-level standard test accuracy in these initial single-seed experiments. "
            f"On arxiv, the teacher scored {arxiv['teacher']:.2f}% and the student-plus-benefit-cache system scored {a['accuracy'] * 100:.2f}%; on products, the corresponding values were {products['teacher']:.2f}% and {p['accuracy'] * 100:.2f}%. "
            "The improvement belongs to the combined system: the frozen student's weights and standalone accuracy did not change."),
        capacity=(f"**Actual cache use in the GIANT study.** Under the 10% node-capacity limit, benefit caching stored {a['cached_nodes']:,} of {arxiv['nodes']:,} arxiv nodes ({100*a['cached_nodes']/arxiv['nodes']:.1f}%) and {p['cached_nodes']:,} of {products['nodes']:,} products nodes ({100*p['cached_nodes']/products['nodes']:.1f}%). "
            "The policy leaves capacity unused when predicted benefit is nonpositive. Cache hits reuse saved teacher outputs; misses run the student, with no online teacher execution."),
        latency=(f"**GIANT serving latency.** For 1,024 GPU-resident queries, median latency increased from {arxiv['policies']['none']['latency_ms_median']:.2f} to {a['latency_ms_median']:.2f} ms on arxiv and from {products['policies']['none']['latency_ms_median']:.2f} to {p['latency_ms_median']:.2f} ms on products. "
            "The accuracy recovery therefore does not establish faster inference than the uncached MLP."),
        weighted=(f"**GIANT request-weighted accuracy.** Under the synthetic Zipf workload, benefit caching changes arxiv from {arxiv['policies']['none']['weighted_accuracy']*100:.2f}% to {a['weighted_accuracy']*100:.2f}%, and products from {products['policies']['none']['weighted_accuracy']*100:.2f}% to {p['weighted_accuracy']*100:.2f}%. "
            "These supplementary results describe the assumed request workload; standard test accuracy is the primary comparison."))


def render_report():
    new = leaderboard_addition()
    citation = json.loads((ROOT / 'results/faith_cache/metrics.json').read_text())
    ogb = json.loads((ROOT / 'results/faith_ogb/metrics.json').read_text())
    assert len(citation) == 690 and len(ogb) == 460
    reference_dir = ROOT / 'results/legacy_no_faith_reference'
    provenance = json.loads((reference_dir / 'provenance.json').read_text())
    legacy = {}
    for dataset, record in provenance['files'].items():
        path = reference_dir / f'{dataset}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        legacy[dataset] = json.loads(path.read_text())

    def group(dataset, policy, variant='frozen'):
        source = ogb if dataset.startswith('ogbn-') else citation
        budget = 0 if policy == 'none' else .1
        result = [r for r in source if r['dataset'] == dataset and r['variant'] == variant
                  and r['policy'] == policy and r['budget'] == budget]
        assert len(result) == 5
        return result

    def score(dataset, policy, metric='accuracy'):
        values = [r[metric] * 100 for r in group(dataset, policy)]
        return f'{S.mean(values):.2f} ± {S.pstdev(values):.2f}'

    results = ['Standard test accuracy (%) is the primary metric: every test node counts equally, as in the OGB leaderboard. Both tables use a 10% node-cache upper bound. The earlier A40 and FAITH studies report five-seed means ± population standard deviations; the GIANT A100 follow-up reports one seed and is preliminary. Request-weighted results are supplementary.', '',
               '## Without FAITH', '',
               'Earlier A40 rows use a two-layer, width-64 GCN teacher and MLP student, with five seeds; benefit caching was not evaluated. GIANT A100 rows use stronger graph teachers and three-layer, width-256 MLP students with precomputed GIANT-XRT features, with one seed. These are separate configurations with different inputs and training protocols, so compare cache policies within each dataset and study.', '',
               '| Dataset | Study | Teacher | Student without cache | Frequency cache | Disagreement cache | Benefit cache |',
               '|---|---|---:|---:|---:|---:|---:|']
    for dataset in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'ogbn-products']:
        teachers = [r['teacher_accuracy'] * 100 for r in legacy[dataset] if r['policy'] == 'kd_no_cache'] if dataset.startswith('ogbn-') else []
        if teachers:
            assert len(teachers) == 5
        teacher = f'{S.mean(teachers):.2f} ± {S.pstdev(teachers):.2f}' if teachers else '—'
        cells = [dataset, 'Earlier A40', teacher]
        for policy in ['kd_no_cache', 'kd_lfu', 'kd_disagreement']:
            budget = 0 if policy == 'kd_no_cache' else .1
            values = [r['accuracy'] * 100 for r in legacy[dataset] if r['policy'] == policy and r['budget'] == budget]
            assert len(values) == 5
            cells.append(f'{S.mean(values):.2f} ± {S.pstdev(values):.2f}')
        results.append('| ' + ' | '.join(cells + ['—']) + ' |')
    results += new['rows'] + ['', new['summary'], '', new['capacity'], '', new['latency']]
    results += ['', '## With FAITH', '',
                'FAITH A100 study: released citation configurations and the documented OGB FAITH adaptation. Architectures, features, teachers, seeds and training budgets differ from the studies without FAITH. Compare cache policies within each configuration; differences between studies do not measure FAITH alone.', '',
                '| Dataset | Teacher | Student without cache | Frequency cache | Disagreement cache | Benefit cache |',
                '|---|---:|---:|---:|---:|---:|']
    for dataset, label in [('cora', 'Cora'), ('citeseer', 'CiteSeer'), ('pubmed', 'PubMed'),
                           ('ogbn-arxiv', 'ogbn-arxiv'), ('ogbn-products', 'ogbn-products')]:
        teachers = [r['teacher_accuracy'] * 100 for r in group(dataset, 'none')] if dataset.startswith('ogbn-') else []
        teacher = f'{S.mean(teachers):.2f} ± {S.pstdev(teachers):.2f}' if teachers else '—'
        results.append('| ' + ' | '.join([label, teacher] + [score(dataset, p) for p in ['none', 'frequency', 'disagreement', 'benefit']]) + ' |')
    products = group('ogbn-products', 'benefit')
    hits = S.mean(r['hit_rate'] for r in products) * 100
    results += ['',
                '**Accuracy benefit.** Benefit caching improves standard accuracy on both OGB datasets by about 4.4 percentage points. This is a gain for the combined student-plus-cache system; the independently distilled student weights are unchanged in this comparison. The large CiteSeer variation includes the failed base seed, which remains in every mean.', '',
                '**FAITH latency and graph smoothness.** OGB benefit-cache query batches take about 0.86–0.87 ms, compared with about 0.45 ms for the uncached student. The measured accuracy gain is accompanied by higher local serving latency. Exact graph unfairness improves slightly for arxiv benefit caching and worsens for products; the accuracy gain does not consistently preserve graph smoothness.', '',
                '## Supplementary workload metrics', '',
                '**FAITH request-weighted accuracy.** Under the synthetic workload, benefit caching changes arxiv accuracy from ' + score('ogbn-arxiv', 'none', 'weighted_accuracy').split(' ± ')[0] + '% to ' + score('ogbn-arxiv', 'benefit', 'weighted_accuracy').split(' ± ')[0] + '%, and products from ' + score('ogbn-products', 'none', 'weighted_accuracy').split(' ± ')[0] + '% to ' + score('ogbn-products', 'benefit', 'weighted_accuracy').split(' ± ')[0] + '%. These are workload-specific metrics, not leaderboard accuracy.', '',
                f'**FAITH request routing.** In the products FAITH benefit-cache run, cached teacher answers cover about {hits:.1f}% of synthetic test requests; the student serves the remaining {100-hits:.1f}%. About 10.3% of test products are cached when test products are counted equally. The teacher therefore does not supply most predictions, and its model does not execute online.', '',
                '**FAITH joint refinement.** Against the equally trained disagreement-cache control at 10%, joint refinement changes request-weighted accuracy by approximately +0.12 points on arxiv and −0.04 points on products. Applying benefit caching to joint rather than independently continued students adds essentially no gain in these OGB runs.', '', new['weighted']]
    text = (ROOT / 'docs/report-template.md').read_text()
    text = text.replace('{{AMAZON_EXAMPLE}}', (ROOT / 'docs/amazon-products-example.md').read_text().strip())
    text = text.replace('{{IMPORTANT_RESULTS}}', '\n'.join(results))
    assert '{{' not in text and text.count('# Important results\n') == 1
    return text.rstrip() + '\n'


def write_current_report():
    text = render_report()
    (ROOT / 'FAITH_CACHE_STUDY.md').write_text(text)
    # Keep the original user-facing local report in sync with the study report.
    mirror = ROOT.parent / 'Cache_Aware_GNN_Distillation_Technical_Report.md'
    if mirror.exists():
        mirrored = re.sub(r'\]\((results/[^)]+)\)', r'](faith-cache-research/\1)', text)
        mirror.write_text(mirrored)
    return text


if __name__ == '__main__':
    write_current_report()
