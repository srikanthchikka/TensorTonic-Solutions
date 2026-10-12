import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    """
    Returns one permutation importance per feature.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    base_score = np.mean(np.asarray(predict_fn(X)) == y) 
    
    # rng = np.random.default_rng(seed=42)
    imps = np.zeros(X.shape[1])
    for idx in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            perm_X = X.copy()
            perm_X[:, idx] = rng.permutation(X[:, idx])
            score = np.mean(np.asarray(predict_fn(perm_X) == y))
            drops.append(base_score - score)
        imps[idx] = np.mean(drops)
    return imps.tolist()


        
        
        
            
        
        
