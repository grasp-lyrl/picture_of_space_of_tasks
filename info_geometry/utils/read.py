import glob
import pickle
import os
import numpy as np
import h5py
import hdf5plugin


def read_preds(path, reindex=False):
    """
    Read stored predictions from h5 file
    """
    if not reindex:
        fn_prefix = "preds_*.h5"
    else:
        fn_prefix = "reindex_preds_*.h5"
    h5_path = os.path.join(path, fn_prefix)

    fnames, preds = [], []
    for h5_file in glob.glob(h5_path):
        fp = h5py.File(h5_file, 'r')
        preds.append(fp['data'])
        fnames.append(h5_file)

    return fnames, preds


def imagenet_start_end():
    """
    Compute p0 and pstar for ImageNet validation dataset
    Make sure that the ordering is the same as your prediction vectors
    """
    labels = np.load("../predictions/labels/imagenet_val_labels.npy")
    p_0 = np.sqrt(np.zeros((50000, 1000)) + 0.001)
    p_star = np.sqrt(np.eye(1000)[labels])
    return p_0, p_star


def write_pred(fname, reindex_pred):
    """
    Write reindexed trajectory to disk
    """
    reindex_pred = np.round(reindex_pred, 4).astype(np.float16)

    fn_suffix = fname.split("/")[-1]
    fn_prefix = '/'.join(fname.split("/")[:-1])
    fname_new = fn_prefix + "/reindex_" + fn_suffix

    with h5py.File(fname_new, 'w') as fp:
        fp.create_dataset('data', data=reindex_pred, **hdf5plugin.LZ4())


def read_progress(fdir, fnames):
    """
    Read progress from pkl file
    """
    with open(os.path.join(fdir, "progress.pkl"), "rb") as fp:
        info = pickle.load(fp)

    progs = []
    for fn in fnames:
        fn = fn.split("/")[-1]
        fn = fn[fn.find("_")+1:]
        progs.append(info[fn]['progress'])

    return progs
