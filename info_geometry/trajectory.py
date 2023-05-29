import numpy as np
import pickle
import os

from tqdm import tqdm
from functools import partial
from scipy import optimize
from utils.read import read_preds, write_pred, read_progress
from utils.read import imagenet_start_end

from utils.plot import print_lengths, plot_trajectory_distances


def dbhat(pr0, pr1):
    """
    Bhattacharyya distance between two models
    """
    pr0 = pr0.astype(np.float32)
    pr1 = pr1.astype(np.float32)

    dcoef = (pr0 * pr1).sum(-1)
    dcoef = np.clip(dcoef, a_min=1e-9, a_max=None)
    dist = -np.log(dcoef).mean(-1)

    return dist


class Geodesic():
    @staticmethod
    def interpolate(start, end, t):
        """
        - Get a point on the geodesic joining "start" and "end". 
        - The geodesic is the same as the geodesic on the surface of a sphere
          which is the great cirlce equation
        """
        cospq = (start * end).sum(-1, keepdims=True)
        dg = np.arccos(np.clip(cospq, 0, 1))

        # Use masks, incase, start and end are identical
        mask = (dg <= 1e-6).reshape(-1)
        gamma = np.array(start)
        gamma[~mask] = np.sin((1-t)* dg[~mask]) * start[~mask] + \
                       np.sin(t    * dg[~mask]) * end[~mask]
        gamma[~mask] = gamma[~mask] / np.sin(dg[~mask])

        return gamma

    @staticmethod
    def project(pred, start, end):
        """
        - Computes the progress (λ) for each model in the trajectory.
        - Progress is a scalar that belongs to the set [0, 1].
        - It measures how far the model is along the geodesic
          joining 'start' and 'end'.
        """
        def dB(t):
            vec = Geodesic.interpolate(start, end, t)
            dist = dbhat(vec, pred)
            return dist

        # Find the point on the geodesic with smallest distance
        # to the model. This corresponds to the geometric progress. 
        lam = optimize.minimize_scalar(dB, bounds=(0, 1), method='bounded').x
        return lam

    @staticmethod
    def progress(trajectory, p_0, p_star):
        targets = np.argmax(p_star, axis=1)
        prog_list, acc_list = [], []

        for ep in tqdm(range(len(trajectory))):
            cur_pr = trajectory[ep].astype(np.float32)

            preds = np.argmax(cur_pr, axis=1)
            prog = Geodesic.project(
                [np.sqrt(cur_pr)], p_0, p_star)

            acc = np.sum(preds == targets) / len(targets)
            prog_list.append(prog)
            acc_list.append(acc)
            
        return prog_list, acc_list


class Trajectory():
    @staticmethod
    def sample(traj, λ_true, λ_sample):
        """
        Sample from the parameterized trajectory
        """
        ind = 0
        max_ind = len(λ_true)

        samples = []

        for λ in tqdm(λ_sample):

            # Find first λ_true (from left) bigger than λ
            while ind < max_ind:
                if λ_true[ind] > λ:
                    break;
                ind += 1

            # Draw sample
            if ind == max_ind:
                sample = traj[-1].astype(np.float32)
            elif ind == 0:
                sample = traj[0].astype(np.float32)
            else:
                seg_start = traj[ind - 1].astype(np.float32)
                seg_end = traj[ind].astype(np.float32)

                λ_start = λ_true[ind-1]
                λ_end = λ_true[ind]
                assert(λ_start <= λ and λ_end >= λ)

                λ_segment = (λ - λ_start) / (λ_end - λ_start)
                λ_segment = np.clip(λ_segment, 0, 1)
                sample = Geodesic.interpolate(seg_start, seg_end, λ_segment)

            samples.append(sample)

        samples = np.array(samples)
        return samples

    @staticmethod
    def reindex(dirs, p_0, p_star, prog_sample):
        """
        Convert a trajectory into a reindexed trajcetory
        sampled at different values of progress (usually uniformly
        spaced values of progress) and store to disk

        Also compute the progress of the original trajectory and
        store to pickle file
        """
        for fdir in dirs:
            fnames, preds = read_preds(fdir)
            info = {}

            for fname, pred in zip(fnames, preds):
                print("Reindeixing trajectories in %s" % fname)
                fn = fname.split("/")[-1]
                prog, acc = Geodesic.progress(pred, p_0, p_star)

                info[fn] = {
                    "progress": prog,
                    "accuracy": acc
                }
                reindex_pred = Trajectory.sample(pred, prog, prog_sample)
                Trajectory.write_pred(fname, reindex_pred)

            with open(os.path.join(fdir, "progress.pkl"), "wb") as fp:
                pickle.dump(info, fp)

    @staticmethod
    def avg_preds(preds, ind):
        """
        Average the predictions at same level of progress
        """
        pr_list = []
        for pr in preds:
            pr_list.append(pr[ind].astype(np.float32))

        return np.mean(pr_list, axis=0)

    @staticmethod
    def compare(dir0, dir1, prog_sample):
        """
        Compare trajectories of reindexed models
        """

        print("Comparing trajectories:\n 1) %s\n 2) %s" % (dir0, dir1))
        dirs = [dir0, dir1]
        preds, progs = [], []

        for fdir in dirs:
            fnames, pr = read_preds(fdir, reindex=True)
            preds.append(pr)
            progs.append(read_progress(fdir, fnames))

        progs = np.array(progs)

        # Get comparable range of pmin/pmax
        pmin = np.max(progs[:, :, 0])
        pmax = np.min(progs[:, :, -1])

        min_ind = np.where(prog_sample >= pmin)[0][0]
        max_ind = np.where(prog_sample <= pmax)[0][-1]

        traj_dist = []
        prog_vals = []
        for ind in tqdm(range(min_ind, max_ind+1)):
            prog_vals.append(prog_sample[ind])

            pr0 = Trajectory.avg_preds(preds[0], ind)
            pr1 = Trajectory.avg_preds(preds[1], ind)

            dist = dbhat(pr0, pr1)
            traj_dist.append(dist)

        return prog_vals, np.array(traj_dist)

    @staticmethod
    def compute_reimann_length(fdir):
        """
        Compute Reimann lengths of a trajectory
        """
        print("Computing Reimann length of:\n%s" % fdir)
        fnames, preds = read_preds(fdir)

        all_lengths = []
        for traj in preds:

            length = 0.0
            for t in tqdm(range(len(traj)-1)):
                dist = dbhat(traj[t], traj[t+1])
                length = length + 2 * np.sqrt(dist)
            all_lengths.append(length)
        all_lengths = np.array(all_lengths)

        return all_lengths.mean(), all_lengths.std()


# Understanding ImageNet-based trajectories
def imagenet_reindex():
    p_0, p_star = imagenet_start_end()
    prog_sample = np.arange(0.0, 1.0+1e-9, 0.02)

    dirs = [
        "../predictions/imagenet_all_v2",
        "../predictions/imagenet_all",
        "../predictions/imagenet_3rd",
        "../predictions/imagenet_instr",
        "../predictions/imagenet_vert"
    ]
    Trajectory.reindex(dirs, p_0, p_star, prog_sample) 


def imagenet_compare_trajectory():
    """
    Compare ImageNet trajectories for 3 tasks
    """
    prog_sample = np.arange(0.0, 1.0+1e-9, 0.02)
    dirs = ["../predictions/imagenet_all",
            "../predictions/imagenet_3rd",
            "../predictions/imagenet_vert"]

    info0 = Trajectory.compare(dirs[0], dirs[1], prog_sample)
    info1 = Trajectory.compare(dirs[0], dirs[2], prog_sample)

    names = ["All vs Random Subset",
             "All vs Vertebrates"]

    plot_name = "../plots/imagenet/trajectories_compare.png"
    plot_trajectory_distances([info0, info1], names, plot_name)


def imagenet_trajectory_lengths():
    """
    Compare ImageNet trajectories for 3 tasks
    """
    dirs = ["../predictions/imagenet_all",
            "../predictions/imagenet_3rd",
            "../predictions/imagenet_vert"]
    dnames = ["All classes",
              "Random subset of 333 classes",
              "Vertebrates"]
    
    len_info = [Trajectory.compute_reimann_length(dr) for dr in dirs]
    print_lengths(len_info, dnames)



if __name__ == '__main__':
    imagenet_reindex()
    imagenet_compare_trajectory()
    imagenet_trajectory_lengths()
