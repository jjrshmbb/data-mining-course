"""严格读取已核查的数据格式，禁止演示或合成数据回退。"""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import ROOT

COMMIT = 'a4963fc976fbb10759717dd6afea51320a5d2639'
DATA_PATH = ROOT / 'data/raw/arrhythmia_imad.npy'
API_URL = ('https://api.github.com/repos/jicongfan/'
           'ImAD-Anomaly-Detection-With-Missing-Data/contents/'
           f'data/Arrhythmia/complete_data.npy?ref={COMMIT}')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def acquire():
    if DATA_PATH.exists():
        return DATA_PATH
    import requests
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(API_URL, timeout=(10, 30))
    r.raise_for_status()
    obj = r.json()
    if obj.get('encoding') != 'base64' or not obj.get('content'):
        raise RuntimeError('GitHub API 未返回文件正文，请检查下载记录。禁止生成替代数据。')
    content = base64.b64decode(obj['content'])
    candidate = DATA_PATH.with_suffix('.download.npy')
    candidate.write_bytes(content)
    a = np.load(candidate, allow_pickle=False)
    if a.shape != (452, 275):
        raise ValueError(f'数据结构变化: {a.shape}')
    candidate.replace(DATA_PATH)
    return DATA_PATH

def load():
    path = acquire()
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    if sha256(path) != manifest['sha256']:
        raise ValueError('原始文件 SHA256 与核查记录不同，请先重新核查数据。')
    a = np.load(path, allow_pickle=False)
    assert a.shape == (452, 275) and a.dtype.kind in 'fi'
    X, y = a[:, :-1].copy(), a[:, -1].astype(int)
    assert np.array_equal(np.unique(y), [0, 1])
    assert np.isfinite(X).all()
    assert X.shape[1] == 274
    return X, y

def split_indices(y, seed):
    # 沿用官方代码的数量规则，不声称恢复论文未发布的固定划分。
    order = np.random.default_rng(seed).permutation(len(y))
    normal = order[y[order] == 0]
    abnormal = order[y[order] == 1]
    ntrain = len(normal) - len(abnormal)
    train = normal[:ntrain]
    test = np.concatenate([normal[ntrain:], abnormal])
    assert not np.intersect1d(train, test).size
    assert len(train) + len(test) == len(y)
    assert np.all(y[train] == 0)
    return train, test

def audit(X, y):
    return {
        'dataset': 'Arrhythmia', 'source_repository_commit': COMMIT,
        'raw_shape': '452 x 275 (last column is label)',
        'n_samples': len(y), 'n_features': X.shape[1],
        'n_normal': int((y == 0).sum()), 'n_anomaly': int((y == 1).sum()),
        'label_meaning': '0 normal / 1 outlier, inherited ODDS binary task',
        'nan_count': int(np.isnan(X).sum()), 'inf_count': int(np.isinf(X).sum()),
        'constant_features_full_data_audit_only': int((np.ptp(X, axis=0) == 0).sum()),
        'duplicate_rows_full_data_audit_only': len(X) - len(np.unique(X, axis=0)),
        'train_rows': 320, 'test_rows': 132, 'test_anomalies': 66,
        'test_prevalence_random_AP_reference': 0.5,
        'split': 'official code count rule, newly seeded indices; no fixed official indices',
        'sha256': sha256(DATA_PATH), 'source_url': API_URL,
    }
