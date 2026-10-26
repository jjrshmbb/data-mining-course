"""检查容易破坏科研结论的错误：划分重叠、掩码复用、评分方向、测试泄漏。"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from research.config import DEFAULT, ROOT
from research.data import load, split_indices
from research.experiment import masked, mask_uniforms, fit_score

def verify():
    checks = []
    X,y = load()
    assert not any(np.array_equal(X[:,i],y) for i in range(X.shape[1]))
    checks.append('274 features exclude final label; no exact label-copy feature')
    train_idx,test_idx=split_indices(y,42)
    assert len(train_idx)==320 and len(test_idx)==132 and y[train_idx].sum()==0
    assert not set(train_idx)&set(test_idx)
    checks.append('normal-only training; split disjoint and exhaustive')
    u=mask_uniforms((320,274),42,100)
    m10=u<.1;m30=u<.3
    assert (m30[m10]).all() and np.array_equal(u,mask_uniforms((320,274),42,100))
    checks.append('seed reproducibility and nested label-independent masks')
    y_small=np.array([0,0,1,1]);scores=np.array([0.,.2,.8,1.])
    assert average_precision_score(y_small,scores)==1 and roc_auc_score(y_small,scores)==1
    checks.append('AP and ROC-AUC use larger-is-more-anomalous direction')
    tr,_=masked(X[train_idx],u,.3)
    te,_=masked(X[test_idx],mask_uniforms((132,274),42,200),.3)
    for imp in ['median','knn']:
        for det in ['iforest','lof']:
            s1,_=fit_score(tr,te,imp,det,42,DEFAULT)
            changed=te.copy();changed[-1,:]=1000000
            s2,_=fit_score(tr,changed,imp,det,42,DEFAULT)
            np.testing.assert_allclose(s1[:-1],s2[:-1],rtol=0,atol=1e-12)
    checks.append('changing one test row cannot change other test-row scores in all 4 workflows')
    latest=ROOT/(ROOT/'reports/latest_run.txt').read_text(encoding='utf8').strip()
    df=pd.read_csv(latest/'results.csv')
    assert df.groupby(['seed','missing_rate']).mask_sha256.nunique().max()==1
    for (seed,det),g in df[df.missing_rate==0].groupby(['seed','detector']):
        if len(g)==2:
            np.testing.assert_allclose(g.AP.iloc[0],g.AP.iloc[1],atol=1e-12)
            np.testing.assert_allclose(g.ROC_AUC.iloc[0],g.ROC_AUC.iloc[1],atol=1e-12)
    checks.append('all methods share identical masks; no-missing imputer results agree')
    assert df[['AP','ROC_AUC']].ge(0).all().all() and df[['AP','ROC_AUC']].le(1).all().all()
    report={'status':'passed','checks':checks,'run':str(latest),'seed':42}
    (latest/'protocol_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    verify()
