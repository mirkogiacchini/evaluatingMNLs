from evaluate.worst_pair import *
from utils.MNL import *
from non_adaptive_algorithm.ilsr import *
from experiments.datasets import *
from evaluate.heuristic_worst_slates import *
from non_adaptive_algorithm.mm import *
from evaluate.average import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from experiments.run_experiments import *
import os
from plotting.plot_results import stylePlot, readResults

def readAvgWorstPair(dsName, dsType, algoName, eps, seeds):
    (n, numqueries,
     avg_pairs, std_pairs,
     avg_worst, std_worst,
     avg_ub,    std_ub,
     avg_full,  std_full,
     avg_mg15, std_mg15,
     avg_noh12, std_noh12,
     avg_avg_pair, std_avg_pair,
     avg_median_pair, std_median_pair) = readResults(dsName, dsType, algoName, eps, seeds, d1_error=True)
    return round(avg_pairs[0], 4)

def readMNLs(dsName, dsType, algoName, queries, seeds=None):
    if seeds is None:
        seeds = list(range(42, 52))
        if algoName == AlgorithmName.MM:
            seeds = [42]

    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName), str(queries)))
    M1 = []
    M2 = []
    for seed in seeds:
        output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))
        with open(output_file, 'r') as f:
            res = json.load(f)
            M1.append(MNL.fromDict(res['M1']))
            M2.append(MNL.fromDict(res['M2']))

    return M1, M2

def plot_two_sampling_schemes_runs(
    ax,
    err_pairs_local_runs,
    err_pairs_global_runs,
    label_local="Local sampling",
    label_global="Global sampling",
    buckets=50,
    show_hist=True,
    title_str='',
    setylabel=True,
):
    """
    Compare two sampling schemes (local vs global),
    where each scheme has multiple randomized runs.

    Parameters
    ----------
    err_pairs_local_runs : list of array-like
        [err_pairs_local_1, ..., err_pairs_local_L]
    err_pairs_global_runs : list of array-like
        [err_pairs_global_1, ..., err_pairs_global_G]
    label_local : str
        Label for local scheme.
    label_global : str
        Label for global scheme.
    buckets : int
        Number of histogram bins.
    show_hist : bool
        Whether to show histograms in addition to KDE curves.
    """
    stylePlot()

    # Pool all runs within each scheme
    err_local_all = np.concatenate(err_pairs_local_runs, axis=0)
    err_global_all = np.concatenate(err_pairs_global_runs, axis=0)

    # Shared x-range across both schemes
    combined_min = min(err_local_all.min(), err_global_all.min())
    combined_max = max(err_local_all.max(), err_global_all.max())

    avg_max_local = np.mean([r.max() for r in err_pairs_local_runs])
    avg_max_global = np.mean([r.max() for r in err_pairs_global_runs])

    # Histogram binning (shared)
    bin_edges = np.linspace(combined_min, combined_max, buckets + 1)

    color_local = '#0072B2' #plt.gca()._get_lines.get_next_color()
    color_global = '#E69F00' #plt.gca()._get_lines.get_next_color()

    runs_per_sampling = [(err_pairs_local_runs, label_local, color_local), 
                        (err_pairs_global_runs, label_global, color_global)]

    # Histograms (normalized to density)
    if show_hist:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Average histogram 
        for runs, label, color in runs_per_sampling:
            local_hists = []
            for r in runs:
                counts, _ = np.histogram(r, bins=bin_edges, density=False)
                local_hists.append(counts / len(r))   # convert to fraction per run
            local_hist_mean = np.mean(local_hists, axis=0)
            avg_density = local_hist_mean / bin_width

            ax.bar(bin_centers, avg_density,
                width=bin_width,
                alpha=0.3, color=color, #label=f"mean histogram ({label})"
                )

    # KDE grid
    x_vals = np.linspace(combined_min, combined_max, 400)

    # ---- KDE: average over runs ----

    for err_sampling_scheme, scheme_label, color_line in runs_per_sampling:
        density_runs = []
        for r in err_sampling_scheme:
            kde_r = gaussian_kde(r)
            density_runs.append(kde_r(x_vals))
        density_runs = np.vstack(density_runs)   # shape (L, len(x_vals))

        density_mean = density_runs.mean(axis=0)
        density_std  = density_runs.std(axis=0)

        ax.plot(
            x_vals,
            density_mean,
            linewidth=2,
            color=color_line,
            label=f"mean KDE ({scheme_label})",
        )

        lower = np.clip(density_mean - density_std, a_min=0.0, a_max=None)
        upper = density_mean + density_std
        ax.fill_between(
            x_vals,
            lower,
            upper,
            color=color_line,
            alpha=0.3,
            #label=f"{scheme_label} KDE ±1 std",
        )
    
    # Vertical lines at max errors (averaged across runs)

    ax.axvline(
        avg_max_local,
        color=color_local,
        linestyle=":",
        linewidth=2,
        label=f"avg. max error ({label_local})",
    )
    ax.axvline(
        avg_max_global,
        color=color_global,
        linestyle="-.",
        linewidth=2,
        label=f"avg. max error ({label_global})",
    )

    # Cosmetics
    ax.set_xlabel("$\ell_1$ Error")
    if setylabel:
        ax.set_ylabel("Density")

    ax.set_title(f"{title_str}")
    ax.grid(alpha=0.2)
    #ax.legend()


def plotDistribution(ax, dsName, dsType, algoName, queries, setylabel, title_str, limit_axis=False):
    seeds = list(range(42, 52))
    if algoName == AlgorithmName.MM:
        seeds = [42]

    offset = 9999
    M1, M2 = readMNLs(dsName, dsType, algoName, queries, seeds=seeds)
    
    local_sampling_errors = []
    global_sampling_errors = []

    for i,seed in enumerate(seeds):
        rng = np.random.default_rng(seed + offset)

        err_pairs_local, num_pairs_considered = computePairwiseErrors(M1[i], M2[i], rng, all_pairs=False, alpha=1.01, K=50)
        err_pairs_sample, _ = computePairwiseErrors(M1[i], M2[i], rng, all_pairs=True, fixed_number_queries=num_pairs_considered)

        local_sampling_errors.append(err_pairs_local)
        global_sampling_errors.append(err_pairs_sample)

    worst_pair = readAvgWorstPair(dsName, dsType, algoName, [queries], seeds)

    plot_two_sampling_schemes_runs(
        ax,
        local_sampling_errors,
        global_sampling_errors,
        label_local="Similar-Weight Pairs",
        label_global="Sampled Pairs",
        buckets=100,
        show_hist=True,
        setylabel=setylabel,
        title_str=f'{title_str}\nworst-pair error: {worst_pair}',
    )

    if limit_axis: 
        if dsName == DatasetName.CHESSCOM:
            if dsType == 'blitz':
                xmax, ymax = 0.25, 170
            else:
                xmax, ymax = 0.25, 190
        elif dsName == DatasetName.CODEFORCES:
            if dsType == 'competing':
                xmax, ymax = 0.8, 280
            elif dsType == 'active':
                xmax, ymax = 0.5, 420
            else:
                xmax, ymax = 0.7, 540
        else:
            if dsType == 'female_elo':
                xmax, ymax = 0.09, 115
            elif dsType == 'female_yelo':
                xmax, ymax = 0.09, 95
            elif dsType == 'male_elo':
                xmax, ymax = 0.09, 100
            else:
                xmax, ymax = 0.09, 95
        
        ax.set_ylim(None, ymax)
        ax.set_xlim(None, xmax)

def plotMultipleDistributions(dsName, dsTypes, algoNames, eps_value, extension='png', limit_axis=False):
    assert len(dsTypes) == 1 or len(algoNames) == 1
    stylePlot()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=max(len(dsTypes), len(algoNames)),
        figsize=(5 * max(len(dsTypes), len(algoNames)), 3.5), #4.5, 3.2 default
        sharey=True  
    )

    idx = 0
    for type in dsTypes:
        for algo in algoNames:
            type_translator = {'female_elo': 'women', 'male_elo':'men', 'female_yelo': 'women-2025', 'male_yelo': 'men-2025'}
            title_str = '' #f'Dataset: {dsName}-{type_translator.get(type, type)}, Algorithm: {algo}'
            ax = axes[idx] if len(dsTypes) * len(algoNames) > 1 else axes
            plotDistribution(ax, dsName, type, algo, eps_value, idx==0, title_str, limit_axis=limit_axis)
            idx += 1

    ax = axes[0] if len(dsTypes) * len(algoNames) > 1 else axes
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, handlelength=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92]) #avoid clashing between legend and plot

    current_file_dir = os.path.dirname(__file__)
    if len(dsTypes) == 1:
        if len(algoNames) > 1:
            output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'plots', f"{dsName}_{dsTypes[0]}", str(eps_value)))
        else:
            output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'plots', f"{dsName}_{dsTypes[0]}", str(algoNames[0]), str(eps_value)))
    else:
        output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'plots', f"{dsName}", str(algoNames[0]), str(eps_value)))
    
    filename = '_'.join([dsName]+dsTypes+algoNames)+"pairs_distribution"
    base_path = os.path.abspath(os.path.join(output_dir, filename))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(base_path+"."+extension)
    plt.close(fig)

if __name__ == '__main__':

    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full')
                ]
    algorithms = [AlgorithmName.ILSR, AlgorithmName.LOBSTER, AlgorithmName.MM]

    eps_value = 0.1

    for extension in ['png', 'pdf']:
        for dsName, dsType in datasets:
            for algo in algorithms:
                print('Plotting: ', dsName, dsType, algo)
                if algo == AlgorithmName.MM and dsName != DatasetName.CODEFORCES:
                    continue 
                if algo == AlgorithmName.ILSR and dsName == DatasetName.CODEFORCES:
                    continue
                plotMultipleDistributions(dsName, [dsType], [algo], eps_value, extension=extension, limit_axis=True)
   