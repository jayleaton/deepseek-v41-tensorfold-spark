#!/usr/bin/env python3
"""The single AB3 analysis (T5-AB3-SPEC.md, sha256 81b34c1a…), frozen before any AB3 request; run once after the last
turn. Reads the frozen layout, every turn's run dir, the tool-result check (ab3_toolcheck.py verify JSON) and TE's
per-turn safety ledger, and writes the verdict: MEETS AB3 / DOES NOT MEET / INCONCLUSIVE.

T (the endpoint) = len(extra.tensorfold.token_ids): the whole returned native id stream, inserted tool results and
forced engine ids once each, prompt excluded. Primary: 100*(sum T_full / sum T_lane - 1) over all 180 pairs.
CI: 20,000 draws, random.Random(20261104); per draw, families in lexicographic order, that family's frozen task count
of task IDs drawn with rng.choice from its sorted IDs, then three seeds drawn with rng.choice for each drawn task; the
same (task, seed) draws serve every contrast; ratio of pooled sums; sorted draws, endpoints [500] and [19499].
Censored = finish deadline/length/timeout, a client close, an error, missing usage or ids, or T >= 32768.
Cut = a runner cell record whose landed ids are fewer than its ids, or whose status is cut.
Usage: python3 -I ab3_analysis.py LAYOUT UNITS TOOLCHECK_JSON LEDGER_JSON OUT_PREFIX RUN_DIR..."""
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

ARMS = {'plain': 'a', 'lane': 'd', 'full': 'rn', 'only': 'ro'}
DRAWS, RNG_SEED, LO, HI = 20000, 20261104, 500, 19499
CAP, NAMESPACE_TOKENS = 32768, 306  # the pinned definition adds 306 prompt tokens (rendered-tool-schema.json)
ORACLE_R7 = '35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc'
IMAGES = ('sha256:26c0b4107869671656476006743be5288dc48c8ee1f5c5b3879bd606fba59535',
          'sha256:1cfb03ebae4f58de4ef61be0bd8527e7d88dd4c046bfce98844bc53305d37292')
ENGINE = '3bf584af1c5407006f154e0c70d05c6f495d0487'


def load(p):
    return json.loads(Path(p).read_text())


def reply(path, grade):
    r = load(path)
    tf = (r.get('extra') or {}).get('tensorfold') or {}
    ids, u = tf.get('token_ids'), r.get('usage') or {}
    comp = tf.get('compute') or {}
    calls = comp.get('calls') or []
    cells = (tf.get('runner') or {}).get('cells') or []
    T = len(ids) if isinstance(ids, list) else None
    integrity = []
    if T is None or r.get('ids_check') != 'match':
        integrity.append(f'ids {r.get("ids_check")}')
    if not u:
        integrity.append('usage missing')
    elif T is not None and T != u.get('completion_tokens', -1) + (comp.get('inserted_tokens') or 0):
        integrity.append('T != completion + inserted')
    if grade is None:
        integrity.append('ungraded')
    censored = (r.get('finish') in ('deadline', 'length', 'timeout') or r.get('closed_by') == 'client'
                or bool(r.get('error')) or not u or T is None or T >= CAP)
    return {'T': T, 'finish': r.get('finish'), 'censored': censored, 'integrity': integrity,
            'solved': None if grade is None else bool(grade.get('ok_signed', grade.get('ok'))),
            'cut': any(c.get('status') == 'cut' or (isinstance(c.get('landed'), int) and isinstance(c.get('ids'), int)
                                                    and c['landed'] < c['ids']) for c in cells),
            'prompt': u.get('prompt_tokens'), 'cached': (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'completion': u.get('completion_tokens'), 'inserted': comp.get('inserted_tokens') or 0,
            'reasoning': (u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
            'calls': len(calls), 'exec_ms': sum(c.get('exec_ms') or 0 for c in calls),
            'executors': [c.get('executor', 'sandbox') for c in calls], 'elapsed_s': r.get('elapsed_s')}


def ratio(rows, a, b, keys):
    sa, sb = sum(rows[a][k]['T'] for k in keys), sum(rows[b][k]['T'] for k in keys)
    return 100 * (sa / sb - 1) if sb else None


def draws(fam_tasks, seeds):
    """The frozen hierarchical draws: a list of DRAWS lists of (task, seed) keys."""
    rng, out = random.Random(RNG_SEED), []
    for _ in range(DRAWS):
        keys = []
        for f in sorted(fam_tasks):
            for _ in range(len(fam_tasks[f])):
                t = rng.choice(fam_tasks[f])
                keys += [(t, rng.choice(seeds)) for _ in range(3)]
        out.append(keys)
    return out


def boot_ci(rows, a, b, D, keep=None):
    vals = []
    for keys in D:
        ks = [k for k in keys if keep is None or k in keep]
        sb = sum(rows[b][k]['T'] for k in ks)
        if sb:
            vals.append(100 * (sum(rows[a][k]['T'] for k in ks) / sb - 1))
    vals.sort()
    return (vals[LO], vals[HI]) if len(vals) == DRAWS else None


def main():
    layout, units = load(sys.argv[1]), [json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines() if x]
    tool, ledger, out, runs = load(sys.argv[3]), load(sys.argv[4]), Path(sys.argv[5]), [Path(p) for p in sys.argv[6:]]
    fam = {u['id']: u['set'] for u in units}
    seeds = tuple(layout['seeds'])
    keys = sorted((t, s) for t in fam for s in seeds)
    fam_tasks = {f: sorted(t for t in fam if fam[t] == f) for f in sorted(set(fam.values()))}
    assert {f: len(v) for f, v in fam_tasks.items()} == layout['k'], 'units are not the layout\'s'
    boots = {b['name']: b for bl in layout['blocks'] for b in bl['boots']}
    rows, problems, safety, integrity = {a: {} for a in ARMS}, [], [], []
    for run in runs:
        grades = {}
        gp = run / 'grades.jsonl'
        for x in (gp.read_text().splitlines() if gp.exists() else []):
            g = json.loads(x)
            grades[(g['arm'], g['id'])] = g
        status = load(run / 'status.json') if (run / 'status.json').exists() else {'attempts': []}
        if status.get('engine_sha') != ENGINE:
            integrity.append(f'{run.name}: engine {status.get("engine_sha")}')
        for att in status['attempts']:
            b = boots.get(att['name'])
            if b is None:
                continue
            arm = b['arm']
            if not (att.get('pass') and att.get('measure_rc') == 0):
                integrity.append(f'{att["name"]}: boot pass {att.get("pass")} measure_rc {att.get("measure_rc")}')
            if att.get('extra') != layout['arms'][arm]['extra']:
                integrity.append(f'{att["name"]}: boot env differs from the frozen config')
            meta = run / att['name'] / 'arm-metadata.json'
            ids = [r.get('image_id') for r in load(meta).get('ranks', [])] if meta.exists() else []
            if tuple(ids) != IMAGES:
                integrity.append(f'{att["name"]}: images {ids}')
            for s in seeds:
                d = run / att['name'] / f'{ARMS[arm]}-s{s}'
                am = d / 'arm.json'
                if not am.exists() or load(am).get('templates_sha256') != b['templates_sha256']:
                    integrity.append(f'{att["name"]}/s{s}: arm.json missing or templates sha differs')
                for t in sorted(fam):
                    f = d / f'{t}.json'
                    if f.exists():
                        rows[arm][(t, s)] = reply(f, grades.get((f'{att["name"]}/{ARMS[arm]}-s{s}', t)))
        wr = run / 'window-report' / 'window-report.json'
        if wr.exists():
            g = load(wr)['gate']
            safety.append({'run': run.name, **{k: g.get(k) for k in ('verdict', 'records', 'ok', 'wrong-value',
                                                                       'wrong-span', 'not-in-text', 'unparsed',
                                                                       'wrong-check', 'cut', 'inconclusive', 'short',
                                                                       'unknown')},
                           'arms': {a['arm']: a.get('oracle') for a in load(wr).get('arms', [])}})
        else:
            safety.append({'run': run.name, 'verdict': 'MISSING'})
    for a in ARMS:
        missing = [k for k in keys if k not in rows[a]]
        if missing:
            integrity.append(f'{a}: {len(missing)} of 180 replies missing, e.g. {missing[:3]}')
        for k, r in rows[a].items():
            integrity += [f'{a} {k[0]}:s{k[1]}: {x}' for x in r['integrity']]
    complete = not integrity and all(len(rows[a]) == 180 for a in ARMS)
    rep = {'spec_sha256': layout['spec_sha256'],
           'layout_sha256': hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
           'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'integrity': integrity,
           'gates': {}, 'arms': {}}
    if not complete:
        rep['verdict'] = 'INCONCLUSIVE'
        rep['why'] = 'missing or unauthenticated evidence (see integrity)'
        return write(out, rep)
    D = draws(fam_tasks, seeds)
    F, L = 'full', 'lane'
    point = ratio(rows, F, L, keys)
    ci = boot_ci(rows, F, L, D)
    per_seed = {s: ratio(rows, F, L, [k for k in keys if k[1] == s]) for s in seeds}
    rep['primary'] = {'change_pct': point, 'ci95': ci, 'per_seed': per_seed,
                      'per_family': {f: {'full': sum(rows[F][k]['T'] for k in keys if fam[k[0]] == f),
                                         'lane': sum(rows[L][k]['T'] for k in keys if fam[k[0]] == f),
                                         'change_pct': ratio(rows, F, L, [k for k in keys if fam[k[0]] == f])}
                                     for f in fam_tasks},
                      'median_task_change_pct': statistics.median(
                          ratio(rows, F, L, [k for k in keys if k[0] == t]) for t in fam),
                      'leave_one_task_out': {t: ratio(rows, F, L, [k for k in keys if k[0] != t]) for t in fam}}
    loo = rep['primary']['leave_one_task_out'].values()
    rep['primary']['leave_one_task_out_range'] = [min(loo), max(loo)]
    sf, sl = sum(rows[F][k]['T'] for k in keys), sum(rows[L][k]['T'] for k in keys)
    seed_sums = [(sum(rows[F][k]['T'] for k in keys if k[1] == s), sum(rows[L][k]['T'] for k in keys if k[1] == s))
                 for s in seeds]
    # exact in integers: point <= -10% is 10*sum_F <= 9*sum_L; each seed's point <= 0 is sum_F <= sum_L
    rep['gates']['token'] = 10 * sf <= 9 * sl and ci[1] <= -5 and all(f <= l for f, l in seed_sums)
    rep['secondary_exploratory'] = {f'{a} vs {b}': {'change_pct': ratio(rows, a, b, keys),
                                                    'ci95': boot_ci(rows, a, b, D)}
                                    for a, b in (('full', 'plain'), ('only', 'lane'), ('full', 'only'),
                                                 ('lane', 'plain'), ('only', 'plain'))}
    # solves: the operational non-inferiority gate on full vs lane
    lost = [k for k in keys if rows[L][k]['solved'] and not rows[F][k]['solved']]
    gained = [k for k in keys if rows[F][k]['solved'] and not rows[L][k]['solved']]
    repeat = sorted(t for t in fam if sum(k[0] == t for k in lost) >= 2)
    diffs = []
    for ks in D:
        diffs.append(sum(rows[F][k]['solved'] - rows[L][k]['solved'] for k in ks) / len(ks))
    diffs.sort()
    rep['solves'] = {'L': len(lost), 'G': len(gained), 'lost': [f'{t}:s{s}' for t, s in lost],
                     'gained': [f'{t}:s{s}' for t, s in gained], 'repeat_loss_tasks': repeat,
                     'paired_diff_task_cluster_ci95': [diffs[LO], diffs[HI]],
                     'audit_all_arms': {a: {'solved': sum(rows[a][k]['solved'] for k in keys),
                                            'losses_vs_lane': sorted(f'{t}:s{s}' for t, s in keys
                                                                     if rows[L][(t, s)]['solved']
                                                                     and not rows[a][(t, s)]['solved'])}
                                        for a in ARMS}}
    rep['gates']['solves'] = len(lost) <= 3 and len(gained) >= len(lost) and not repeat
    # censoring
    cen = {a: [k for k in keys if rows[a][k]['censored']] for a in ARMS}
    extra_by_seed = {s: sum(k[1] == s for k in cen[F]) - sum(k[1] == s for k in cen[L]) for s in seeds}
    both = [k for k in keys if not rows[F][k]['censored'] and not rows[L][k]['censored']]
    no_cut = [k for k in keys if not rows[F][k]['cut'] and not rows[L][k]['cut']]
    s_i = {k: rows[L][k]['T'] - rows[F][k]['T'] for k in keys}
    cpair = [k for k in keys if rows[F][k]['censored'] or rows[L][k]['censored']]
    den, den_pos = sum(s_i.values()), sum(max(v, 0) for v in s_i.values())
    share = sum(s_i[k] for k in cpair) / den if den > 0 else None
    share_pos = sum(max(s_i[k], 0) for k in cpair) / den_pos if den_pos > 0 else None
    both_point = ratio(rows, F, L, both) if both else None
    rep['censoring'] = {'censored': {a: len(v) for a, v in cen.items()}, 'full_minus_lane_by_seed': extra_by_seed,
                        'complete_both_n': len(both), 'complete_both_change_pct': both_point,
                        'complete_both_ci95': boot_ci(rows, F, L, D, set(both)) if both else None,
                        'no_cut_n': len(no_cut), 'no_cut_change_pct': ratio(rows, F, L, no_cut) if no_cut else None,
                        'censored_share_of_savings': share, 'censored_share_of_positive_savings': share_pos}
    rep['gates']['censoring'] = (all(len(v) <= 18 for v in (cen[F], cen[L])) and len(cen[F]) - len(cen[L]) <= 3
                                 and all(v <= 1 for v in extra_by_seed.values()) and both_point is not None
                                 and both_point <= -5 and rep['censoring']['no_cut_change_pct'] is not None
                                 and rep['censoring']['no_cut_change_pct'] <= 0 and share is not None
                                 and share <= 0.25 and share_pos is not None and share_pos <= 0.25)
    # safety: r7 (pinned) on every run, the per-arm cut rule, and every delivered tool result re-executed
    r7_ok = all(x.get('verdict') == 'PASS' and x.get('wrong-value') == 0 and x.get('wrong-span') == 0
                and not any(x.get(k) for k in ('not-in-text', 'unparsed', 'wrong-check', 'inconclusive', 'unknown'))
                for x in safety)
    pinned = all(ledger.get(x['run'], {}).get('oracle_sha_before') == ORACLE_R7 ==
                 ledger.get(x['run'], {}).get('oracle_sha_after') for x in safety)
    cut_rate = {}
    for x in safety:
        for arm, o in (x.get('arms') or {}).items():
            c = cut_rate.setdefault(arm.split(':')[0], [0, 0])
            c[0] += (o or {}).get('cut', 0)
            c[1] += sum(v for k, v in (o or {}).items() if k != 'text_owned' and isinstance(v, int))
    cuts_ok = all(c <= 0.01 * n for c, n in cut_rate.values())
    tool_overall = {k: v.get('overall') for k, v in (tool.get('arms') or {}).items()}
    tool_ok = tool.get('overall') == 'PASS' or (tool_overall and all(v == 'PASS' for v in tool_overall.values()))
    rep['safety'] = {'r7_runs': safety, 'r7_pinned_every_turn': pinned, 'cut_rate': cut_rate,
                     'tool_check': {'overall': tool.get('overall'), 'arms': tool_overall,
                                    'checker_sha256': tool.get('checker_sha256')}}
    rep['gates']['safety'] = r7_ok and pinned and cuts_ok and tool_ok
    tool_inconclusive = any(v == 'INCONCLUSIVE' for v in tool_overall.values()) or tool.get('overall') == 'INCONCLUSIVE'
    # integrity and execution: namespace pinned in every tool reply, cache, wall and tool time per arm
    prompt_bad = [f'{t}:s{s}' for t, s in keys
                  if not (rows['lane'][(t, s)]['prompt'] == rows['only'][(t, s)]['prompt']
                          == rows['plain'][(t, s)]['prompt'] + NAMESPACE_TOKENS)]
    guide = {rows['full'][k]['prompt'] - rows['lane'][k]['prompt'] for k in keys}
    rep['execution'] = {'namespace_prompt_mismatches': prompt_bad, 'guide_prompt_tokens': sorted(guide),
                        'ledger': ledger}
    rep['gates']['provenance'] = not prompt_bad and len(guide) == 1 and all(
        v.get('fence') == 'on' for v in ledger.values())
    for a in ARMS:
        v = [rows[a][k] for k in keys]
        rep['arms'][a] = {'T': sum(x['T'] for x in v), 'model_sampled': sum(x['completion'] for x in v),
                          'inserted_results': sum(x['inserted'] for x in v),
                          'reasoning': sum(x['reasoning'] or 0 for x in v), 'prompt': sum(x['prompt'] for x in v),
                          'cached_prompt': sum(x['cached'] or 0 for x in v), 'solved': sum(x['solved'] for x in v),
                          'censored': sum(x['censored'] for x in v), 'cut_replies': sum(x['cut'] for x in v),
                          'deadlines': sum(x['finish'] in ('deadline', 'timeout') for x in v),
                          'wall_s_sum': round(sum(x['elapsed_s'] for x in v), 1),
                          'wall_s_median': statistics.median(x['elapsed_s'] for x in v),
                          'tool_calls': sum(x['calls'] for x in v), 'tool_exec_s': round(sum(x['exec_ms'] for x in v)
                                                                                      / 1000, 1),
                          'executors': {e: sum(x['executors'].count(e) for x in v)
                                        for e in sorted({e for x in v for e in x['executors']})},
                          'per_family_T': {f: sum(rows[a][k]['T'] for k in keys if fam[k[0]] == f) for f in fam_tasks},
                          'per_seed_T': {s: sum(rows[a][k]['T'] for k in keys if k[1] == s) for s in seeds}}
    g = rep['gates']
    if tool_inconclusive or not pinned:
        rep['verdict'] = 'INCONCLUSIVE'
    elif all(g.values()):
        rep['verdict'] = 'MEETS AB3'
    else:
        rep['verdict'] = 'DOES NOT MEET'
    return write(out, rep)


def write(out, rep):
    Path(f'{out}.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    p, md = rep.get('primary') or {}, [f'# AB3 analysis: **{rep["verdict"]}**', '']
    if p:
        md += [f'Primary (tool-full vs lane, 180 pairs): {p["change_pct"]:+.2f}% [{p["ci95"][0]:+.2f}, '
               f'{p["ci95"][1]:+.2f}]; per seed {", ".join(f"{s}: {v:+.2f}%" for s, v in p["per_seed"].items())}', '']
    md += [f'Gates: {rep["gates"]}', '', f'Integrity problems: {len(rep["integrity"])}']
    md += [f'- {x}' for x in rep['integrity'][:30]]
    Path(f'{out}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:6]))


if __name__ == '__main__':
    main()
