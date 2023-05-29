import pickle
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.read import read_preds
from utils.plot import plot_inpca


def compute_inpca(mat):
    """
    Description:
        compute InPCA on a prediction matrix
    Args:
        mat:
            np.array of shape (num_models x samples x classes)
    Return:
        (singular_val, imaginary):
            Singular values and indicatory if singular value is imaginary
        projection:
            The projections onto each eigen-vector scaled by singular values.
            projection[:, i] gives the ith vector.
    """
    mat = np.sqrt(mat)
    mat1 = np.transpose(mat, axes=[1, 0, 2])
    mat2 = np.transpose(mat, axes=[1, 2, 0])

    Dmat = 0.0
    dim = len(mat1)
    batch = 500
    for i in range(0, dim, batch):
        Dmat += -(np.log(mat1[i:i+batch] @ mat2[i:i+batch])).sum(0)

    # Normalization
    Dmat = Dmat / dim
    Dmat = Dmat / 2

    ldim = Dmat.shape[0]
    Pmat = np.eye(ldim) - 1.0/ ldim
    Wmat = - Pmat @ Dmat @ Pmat

    # Factor of 2 adjustment
    Wmat = Wmat / 2

    eigenval, eigenvec = np.linalg.eigh(Wmat)

    #Sort eigen-values by magnitude
    sort_ind = np.argsort(-np.abs(eigenval))
    eigenval = eigenval[sort_ind]
    eigenvec = eigenvec[:, sort_ind]

    # Find projections
    singular_val = np.sqrt(np.abs(eigenval))
    imaginary = np.array(eigenval < 0.0)
    projection = eigenvec * singular_val.reshape(1, -1)

    return (singular_val, imaginary), projection


def compute_inpca_h5(mats, batch_size=1000):
    """
    Description:
        compute InPCA on a prediction list of prediction matrices
    Args
        mats:
            List of list of prediction vectors [[..], [..], ...]
            Each sub-list is a list of length "number of seeds"
            Each element in predictions for one trajectory
        ncls:
            Number of classes
        batch_size:
            Number of samples to handle at the same time
    Return:
        (singular_val, imaginary):
            Singular values and indicatory if singular value is imaginary
        projection:
            The InPCA embedding vector of shape [num-models x seed x checkpoint x embed dim]

    """
    nsamples, ncls = mats[0][0].shape[1:]

    allmats = [pr for smat in mats for pr in smat]
    nmodels = [[pr.shape[0] for pr in smat] for smat in mats]

    Dmat = 0.0
    for i in tqdm(range(0, 50000, batch_size)):
        mat = []
        for j in range(len(allmats)):
            matj = allmats[j][:, i:i+batch_size].astype(np.float32)
            matj = np.transpose(np.sqrt(matj), axes=[1, 0, 2])
            mat.append(matj)

        mat1 = np.concatenate(mat, axis=1)
        mat2 = np.transpose(mat1, axes=[0, 2, 1])

        dcoef = np.clip(mat1 @ mat2, a_min=1e-6, a_max=None)

        Dmat += -np.log(dcoef).sum(0)
        del mat, mat1, mat2, dcoef
    
    # Compute eigen-decomposition of double-centered distance matrix
    ndim = len(Dmat)
    Pmat = np.eye(ndim) - 1.0/ ndim
    Wmat = - 0.5 * (Pmat @ Dmat @ Pmat)

    eigenval, eigenvec = np.linalg.eigh(Wmat)
    sort_ind = np.argsort(-np.abs(eigenval))
    eigenval = eigenval[sort_ind]
    eigenvec = eigenvec[:, sort_ind]

    # Find InPCA Embedding
    singular_val = np.real(np.sqrt(np.abs(eigenval)))
    imaginary = np.array(eigenval < 0.0)
    projection = np.real(eigenvec * singular_val.reshape(1, -1))

    # Keep only top 3 dimensions (we usually only plot 3)
    projection = projection[:, :3]

    # Reshape projection matrix into
    # [ task x seed x checkpoint x embed dim]
    all_projections = []
    cur = 0
    for ncount in nmodels:
        nxt = sum(ncount)
        nseeds = len(ncount)
        proj = projection[cur:cur+nxt]
        proj = proj.reshape(nseeds, ncount[0] , 3)
        all_projections.append(proj)
        cur += nxt

    return (eigenval, imaginary), all_projections


def imagenet_inpca():
    """
    Create InPCA embedding for trajectory of 3 tasks
    """
    # Fetch prediction vectors
    preds_im = read_preds("../predictions/imagenet_all")[1]
    preds_3rd = read_preds("../predictions/imagenet_3rd")[1]
    preds_instr = read_preds("../predictions/imagenet_instr")[1]
    preds_vert = read_preds("../predictions/imagenet_vert")[1]

    # Compute InPCA embedding
    all_preds = [preds_im, preds_3rd, preds_instr, preds_vert]
    eigen, embedding = compute_inpca_h5(all_preds, batch_size=1000)

    # Compute explained stress
    eig_sq = eigen[0] ** 2
    expl_stress = 1 - np.sqrt(1 - np.sum(eig_sq[0:3]) / np.sum(eig_sq))
    print("Explained stress is first 3 dimensions %0.2f%%" % (expl_stress * 100))

    # Store inpca embedding to file
    fname = "../predictions/embedding/imagenet_inpca.pkl"
    fdir = "/".join(fname.split("/")[:-1])

    obj = {
        "singular_value": eigen[0],
        "imaginary": eigen[1],
        "inpca_embedding": embedding
    }
    os.makedirs(fdir, exist_ok=True)
    with open(fname , "wb") as fp:
        pickle.dump(obj, fp)


def imagenet_plot_inpca():

    fname = "../predictions/embedding/imagenet_inpca.pkl"
    with open(fname , "rb") as fp:
        obj = pickle.load(fp)
    embed = obj['inpca_embedding']

    # Crete table of values
    name = {0: "Imagenet all",
            1: "Imagenet Random subset",
            2: "Vertebrates",
            3: "Instrumentality"}

    cols = {0: 'lightcoral',
            1: 'pink', 
            2: '#66c2a5',
            3: '#3288bd'}

    tab = []
    for i in range(len(embed)):
        for s in range(embed[i].shape[0]):
            for t in range(embed[i].shape[1]):
                x = list(embed[i][s, t, :3])
                x = x + [t, name[i], s]
                tab.append(x)

    tab = pd.DataFrame(tab)
    tab.columns = ['PC 1', 'PC 2', 'PC 3', 'Epoch', "Model", "Seed"]
    tab.head()

    fname = "../plots/imagenet/inpca.html"
    plot_inpca(tab, name, cols, fname)


if __name__ == '__main__':
    # imagenet_inpca()
    imagenet_plot_inpca()
