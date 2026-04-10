import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_distances


def knn_graph(
        X: ArrayLike,
        metric: str = 'cosine',  
        k: int = 20,
        mutual: bool = True,
        verbose: bool = True
) -> ArrayLike:
    """
    Args:
        X (ArrayLike): input data (rows are samples)
        metric (str, optional): metric for distance (see sklearn.metrics) Defaults to 'cosine'.
        k (int, optional): neighbors. Defaults to 20.
        mutual (bool, optional): if mutual knn
    Returns:
        ArrayLike: knn_graph as a matrix, with values in {0,1}. 
    """
    if verbose: print("Building graph...")
    dist_mat = pairwise_distances(X, metric=metric)
    
    # self-loops will not be considered
    np.fill_diagonal(dist_mat,np.inf) 
    # argpartition takes smallest k elements and put them in the first k pos.
    k_nearest = np.argpartition(dist_mat,k)[:,:k]

    A = np.zeros_like(dist_mat)
    np.put_along_axis(A,k_nearest,1, axis=-1)
    
    if mutual:
        if verbose: 
            print(f"Disconnected components: {np.sum(np.sum(A * A.T, axis=1) == 0)}")         
        return (A * A.T)
    else:
        if verbose: 
            print(f"Disconnected components: {np.sum(np.sum(A, axis=1) == 0)}")         
        return A





















