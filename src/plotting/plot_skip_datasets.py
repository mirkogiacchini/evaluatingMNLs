from plotting.plot_algo_comparison import readResults1vs1
from experiments.run_experiments import AlgorithmName, DatasetName
from plotting.plot_results import stylePlot, readResults
import matplotlib.pyplot as plt
import os 

def plotMultipleDsSizes(skiptype, sizes, eps, d1_error=True, extension='png', plot_differences = True):
    stylePlot()
    pairs_as, pairs_std_as = [], []
    worst_as, worst_std_as = [], []
    ub_as, ub_std_as = [], []
    full_as, full_std_as = [], []
    avg_pair_as, avg_pair_std_as = [], []

    pairs2_as, pairs2_std_as = [], []

    algoName1 = AlgorithmName.ILSR
    algoName2 = AlgorithmName.LOBSTER

    seeds = list(range(42, 42+10))
    for size in sizes:
        if plot_differences:
            (n, numqueries,
            avg_pairs, std_pairs,
            avg_worst, std_worst,
            avg_ub,    std_ub,
            avg_full,  std_full,
            avg_mg15, std_mg15,
            avg_noh12, std_noh12,
            avg_avg_pair, std_avg_pair,
            avg_median_pair, std_median_pair) = readResults1vs1(DatasetName.SKIP, f'{skiptype}#{size}', algoName1, algoName2, [eps], seeds, d1_error=d1_error)
            assert n == size

            pairs_as.append(avg_pairs[0])
            pairs_std_as.append(std_pairs[0])

            worst_as.append(avg_worst[0])
            worst_std_as.append(std_worst[0])
        
            ub_as.append(avg_ub[0])
            ub_std_as.append(std_ub[0])
        
            full_as.append(avg_full[0])
            full_std_as.append(std_full[0])

            avg_pair_as.append(avg_avg_pair[0])
            avg_pair_std_as.append(std_avg_pair[0])
        else:
            (n, numqueries,
            avg_pairs1, std_pairs1,
            avg_worst1, std_worst1,
            avg_ub1,    std_ub1,
            avg_full1,  std_full1,
            avg_mg151, std_mg151,
            avg_noh121, std_noh121,
            avg_avg_pair1, std_avg_pair1,
            avg_median_pair1, std_median_pair1) = readResults(DatasetName.SKIP, f'{skiptype}#{size}', algoName1, [eps], seeds, d1_error=d1)

            (n2, numqueries2,
            avg_pairs2, std_pairs2,
            avg_worst2, std_worst2,
            avg_ub2,    std_ub2,
            avg_full2,  std_full2,
            avg_mg152, std_mg152,
            avg_noh122, std_noh122,
            avg_avg_pair2, std_avg_pair2,
            avg_median_pair2, std_median_pair2) = readResults(DatasetName.SKIP, f'{skiptype}#{size}', algoName2, [eps], seeds, d1_error=d1)

            assert n == n2 == size and numqueries == numqueries2

            pairs_as.append(avg_pairs1[0])
            pairs_std_as.append(std_pairs1[0])
            pairs2_as.append(avg_pairs2[0])
            pairs2_std_as.append(std_pairs2[0])

    fig, ax = plt.subplots()

    nameRefactor = {AlgorithmName.ILSR_RS: 'ilsr (repeat samples)'}

    # Define series with labels and markers
    if plot_differences:
        series = [
            ("Pairs",                    pairs_as, pairs_std_as, "o"),
            ("Lower Bound (Worst Slate)", worst_as, worst_std_as, "s"),
            ("Upper Bound",              ub_as,    ub_std_as,    "^"),
            #("Full-Slate",               full_as,  full_std_as,  "D"),
            #("Full-Slate (RMSE)",               avg_mg15,  std_mg15,  "v"),
            #("Full-Slate (normalized $\ell_2$)",   avg_noh12,  std_noh12,  "P"),
            #("Avg Pairs",                    avg_pair_as, avg_pair_std_as, "H"),
            #("Median Pairs",                    avg_median_pair, std_median_pair, "*"),
        ]
    else:
        print(pairs2_std_as)
        series = [
            (f"Pairs ({nameRefactor.get(algoName1, algoName1)})", pairs_as, pairs_std_as, "o"),
            (f"Pairs ({nameRefactor.get(algoName2, algoName2)})", pairs2_as, pairs2_std_as, "s"),
        ]

    for label, avg, std, marker in series:
        ax.errorbar(
            sizes,
            avg,
            yerr=std,
            label=label,
            marker=marker,
            #linestyle="-",
            capsize=2,
            elinewidth=.8,
            linewidth=1, #1.5
        )

    #ax.set_xscale("log")

    if plot_differences:
        ymin = max(-2, min(min(pairs_as), min(worst_as), min(ub_as), min(full_as), min(avg_pair_as)) - max(max(pairs_std_as), max(worst_std_as), max(ub_std_as), max(full_std_as), max(avg_pair_std_as))*1.1)
        #ymax = min(2.1, (max(avg_worst) + max(std_worst)) * 1.1)
        ymax = min(2.1, (max(ub_as) + max(ub_std_as)) * 1.1)
    else:
        ymin = 0
        ymax = min(2.1, max(max(pairs_as) + max(pairs_std_as), max(pairs2_as) + max(pairs2_std_as)) * 1.1)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Number of Items ($n$)")
    if d1:
        if plot_differences:
            ax.set_ylabel(f"$\ell_1$ difference: {algoName1} $-$ {algoName2}")
        else:
            ax.set_ylabel("$\ell_1$-error")
    else:
        if plot_differences:
            ax.set_ylabel(f"$\ell_\infty$ difference: {algoName1} $-$ {algoName2}")
        else:
            ax.set_ylabel("$\ell_\infty$-error")
    dsName, dsType = skiptype.split('#')

    #ax.set_title(f"Dataset: {dsName}-{dsType}")
    ax.legend(loc="best")

    fig.tight_layout()

    current_file_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'plots',f"{dsName}_{dsType}", 'skip_exp'))
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.abspath(os.path.join(output_dir, f'{dsName}_{dsType}_{algoName1}_vs_{algoName2}_'+('d1' if d1 else 'dinfty')))

    fig.savefig(f"{base_path}.{extension}")

    plt.close(fig)


if __name__ == '__main__':
    skip_types= ['chess.com#bullet',
                   'chess.com#blitz',
                   'codeforces#competing',
                   'codeforces#active',
                   'codeforces#full',
                   ]
    step_size, max_size = 250, 5000
    sizes = [i for i in range(step_size, max_size+1, step_size)]

    eps = 0.1

    for extension in ['png', 'pdf']:
        for d1 in [True, False]:
            for dstype in skip_types:
                print(f'plotting {dstype}', d1, extension)
                plotMultipleDsSizes(dstype, sizes, eps, d1_error=d1, extension=extension, plot_differences=False)