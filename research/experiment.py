import copy
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from .config import ROOT, DEFAULT
from .data import load, audit, split_indices

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def mask_uniforms(shape, seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream])).random(shape)

def masked(X, uniforms, rate):
    mask = uniforms < rate
    result = X.copy()
    result[mask] = np.nan
    return result, mask

def fit_score(train_missing, test_missing, imputer_name, detector_name, seed, config):
    # StandardScaler 忽略 NaN，且只在训练集拟合。
    # 先缩放再做 KNN，避免大数值量纲主导距离。两种填补使用同一顺序。
    start = time.perf_counter()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_missing)
    test_scaled = scaler.transform(test_missing)
    if imputer_name == 'median':
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    elif imputer_name == 'knn':
        imputer = KNNImputer(n_neighbors=config['knn_neighbors'], weights='uniform', keep_empty_features=True)
    else:
        raise ValueError(imputer_name)
    train = imputer.fit_transform(train_scaled)
    test = imputer.transform(test_scaled)
    assert np.isfinite(train).all() and np.isfinite(test).all()
    preprocess_seconds = time.perf_counter() - start
    if detector_name == 'iforest':
        model = IsolationForest(n_estimators=config['iforest_n_estimators'], max_samples='auto',
                                contamination='auto', random_state=seed, n_jobs=config['n_jobs'])
    elif detector_name == 'lof':
        model = LocalOutlierFactor(n_neighbors=config['lof_neighbors'], novelty=True,
                                  contamination='auto', n_jobs=config['n_jobs'])
    else:
        raise ValueError(detector_name)
    start_fit = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start_fit
    start_score = time.perf_counter()
    # 两种模型 score_samples 越小越异常，统一反号。
    # LOF novelty=True 仅对从未用于拟合的新测试样本调用 score_samples。
    scores = -model.score_samples(test)
    score_seconds = time.perf_counter() - start_score
    assert len(scores) == len(test) and np.isfinite(scores).all()
    return scores, {'preprocess_seconds': preprocess_seconds, 'fit_seconds': fit_seconds,
                    'score_seconds': score_seconds,
                    'total_seconds': preprocess_seconds + fit_seconds + score_seconds}

def run(config=None, label='baseline'):
    config = copy.deepcopy(DEFAULT if config is None else config)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'reports/runs' / f'{stamp}_{label}_{uuid.uuid4().hex[:6]}'
    out.mkdir(parents=True)
    for sub in ['masks', 'splits', 'predictions', 'figures']:
        (out / sub).mkdir()
    logger = logging.getLogger(str(out))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in [logging.FileHandler(out / 'run.log', encoding='utf-8'), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    dump(out / 'config.json', config)
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'timestamp_Asia_Shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
           'packages': {name: importlib.metadata.version(name) for name in
                        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'requests']}}
    try:
        env['base_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        env['branch'] = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    except Exception as exc:
        env['git_error'] = repr(exc)
    # 保存真正参与实验的源码散列，覆盖“未提交改动无法复查”的情况。
    env['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'research').glob('*.py'))}
    dump(out / 'environment.json', env)
    rows, failures = [], []
    (out / 'failures.jsonl').write_text('', encoding='utf-8')
    try:
        X, y = load()
        data_audit = audit(X, y)
        pd.DataFrame([data_audit]).to_csv(out / 'data_audit.csv', index=False, encoding='utf-8-sig')
        dump(out / 'data_audit.json', data_audit)
        if config.get('warmup'):
            warm_train, warm_test = split_indices(y, config['seeds'][0])
            for detector in config['detectors']:
                fit_score(X[warm_train], X[warm_test], 'median', detector, config['seeds'][0], config)
            logger.info('Warmup complete, scores discarded and excluded from result table/timing.')
        for seed in config['seeds']:
            train_idx, test_idx = split_indices(y, seed)
            np.savez_compressed(out / 'splits' / f'seed_{seed}.npz', train_indices=train_idx, test_indices=test_idx)
            train_X, test_X, test_y = X[train_idx], X[test_idx], y[test_idx]
            utr = mask_uniforms(train_X.shape, seed, 100)
            ute = mask_uniforms(test_X.shape, seed, 200)
            for rate in config['missing_rates']:
                tr, mtr = masked(train_X, utr, rate)
                te, mte = masked(test_X, ute, rate)
                mask_id = hashlib.sha256(mtr.tobytes() + mte.tobytes()).hexdigest()
                tag = f'seed_{seed}_rate_{round(rate*100):02d}'
                np.savez_compressed(out / 'masks' / f'{tag}.npz', train=mtr, test=mte)
                for imp in config['imputers']:
                    for det in config['detectors']:
                        context = {'seed': seed, 'missing_rate': rate, 'imputer': imp, 'detector': det}
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                scores, timing = fit_score(tr, te, imp, det, seed, config)
                            row = {**context, 'AP': average_precision_score(test_y, scores),
                                   'ROC_AUC': roc_auc_score(test_y, scores), **timing,
                                   'train_missing_actual': mtr.mean(), 'test_missing_actual': mte.mean(),
                                   'mask_sha256': mask_id, 'train_n': len(tr), 'test_n': len(te),
                                   'test_positive_fraction': test_y.mean(),
                                   'warnings': ' / '.join(str(w.message) for w in caught)}
                            rows.append(row)
                            pd.DataFrame({'original_row': test_idx, 'y_for_evaluation_only': test_y,
                                          'anomaly_score': scores}).to_csv(
                                out / 'predictions' / f'{tag}_{imp}_{det}.csv', index=False)
                            pd.DataFrame(rows).to_csv(out / 'results.csv', index=False, encoding='utf-8-sig')
                            logger.info('%s AP=%.4f ROC_AUC=%.4f seconds=%.3f', context,
                                        row['AP'], row['ROC_AUC'], row['total_seconds'])
                        except Exception as exc:
                            failure = {**context, 'error': repr(exc), 'traceback': traceback.format_exc()}
                            failures.append(failure)
                            with (out / 'failures.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(failure, ensure_ascii=False) + '\n')
                            logger.exception('workflow failed: %s', context)
        if not rows:
            raise RuntimeError('所有流程失败，请查看 failures.jsonl。')
        df = pd.DataFrame(rows)
        group = ['imputer', 'detector', 'missing_rate']
        summary = df.groupby(group).agg(n=('AP', 'size'), AP_mean=('AP', 'mean'), AP_std=('AP', 'std'),
            ROC_AUC_mean=('ROC_AUC', 'mean'), ROC_AUC_std=('ROC_AUC', 'std'),
            total_seconds_mean=('total_seconds', 'mean')).reset_index()
        summary.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
        base = df[df.missing_rate == 0][['seed','imputer','detector','AP','ROC_AUC']]
        paired = df.merge(base, on=['seed','imputer','detector'], suffixes=('', '_complete'))
        paired['delta_AP_vs_complete'] = paired.AP - paired.AP_complete
        paired['delta_ROC_AUC_vs_complete'] = paired.ROC_AUC - paired.ROC_AUC_complete
        paired.to_csv(out / 'paired_differences.csv', index=False, encoding='utf-8-sig')
        from .plotting import plots, analysis
        plots(df, out)
        analysis(df, paired, out)
        dump(out / 'status.json', {'status': 'complete' if not failures else 'partial',
                                  'completed_workflows': len(rows), 'failed_workflows': len(failures)})
        (ROOT / 'reports/latest_run.txt').write_text(str(out.relative_to(ROOT)), encoding='utf-8')
        logger.info('Finished: %s', out)
        return out
    except Exception:
        logger.exception('Run aborted')
        dump(out / 'status.json', {'status': 'aborted', 'traceback': traceback.format_exc()})
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
