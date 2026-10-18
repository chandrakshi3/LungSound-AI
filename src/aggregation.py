import numpy as np
def aggregate_scores(scores,threshold=.5,method='max'):
    a=np.asarray(scores,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not a.size: raise ValueError('scores must be a non-empty Nx2 array')
    if not np.all(np.isfinite(a)) or np.any((a<0)|(a>1)): raise ValueError('scores must be finite and in [0,1]')
    if method=='max': agg=a.max(axis=0)
    elif method=='mean': agg=a.mean(axis=0)
    else: raise ValueError('method must be max or mean')
    c,w=bool(agg[0]>=threshold),bool(agg[1]>=threshold)
    return {'crackles_score':float(agg[0]),'wheezes_score':float(agg[1]),'crackles_predicted':c,'wheezes_predicted':w,'category':category(c,w),'aggregation':method,'threshold':threshold}
def category(c,w): return 'Both' if c and w else 'Crackles' if c else 'Wheezes' if w else 'Normal'
