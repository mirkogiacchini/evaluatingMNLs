from plotting.plot_algo_comparison import readResults1vs1
from experiments.run_experiments import AlgorithmName, DatasetName
from plotting.plot_results import stylePlot, readResults
import matplotlib.pyplot as plt
import os 

def plotMultipleAlphas(alphas, size, eps, d1_error=True, extension='png'):
    stylePlot()
    d1 = d1_error
    pairs_as, pairs_std_as = [], []
    worst_as, worst_std_as = [], []
    ub_as, ub_std_as = [], []
    full_as, full_std_as = [], []
    avg_pair_as, avg_pair_std_as = [], []

    pairs2_as, pairs2_std_as = [], []
    ub2_as, ub2_std_as = [], []
    worst2_as, worst2_std_as = [], []
    
    algoName1 = AlgorithmName.ILSR
    algoName2 = AlgorithmName.LOBSTER

    seeds = list(range(42, 42+10))
    for alpha in alphas:
        (n, numqueries,
        avg_pairs1, std_pairs1,
        avg_worst1, std_worst1,
        avg_ub1,    std_ub1,
        avg_full1,  std_full1,
        avg_mg151, std_mg151,
        avg_noh121, std_noh121,
        avg_avg_pair1, std_avg_pair1,
        avg_median_pair1, std_median_pair1) = readResults(DatasetName.POWERLAW, f'{alpha}#{size}', algoName1, [eps], seeds, d1_error=d1_error)

        (n2, numqueries2,
        avg_pairs2, std_pairs2,
        avg_worst2, std_worst2,
        avg_ub2,    std_ub2,
        avg_full2,  std_full2,
        avg_mg152, std_mg152,
        avg_noh122, std_noh122,
        avg_avg_pair2, std_avg_pair2,
        avg_median_pair2, std_median_pair2) = readResults(DatasetName.POWERLAW, f'{alpha}#{size}', algoName2, [eps], seeds, d1_error=d1_error)

        assert n == n2 == size and numqueries == numqueries2

        pairs_as.append(avg_pairs1[0])
        pairs_std_as.append(std_pairs1[0])
        pairs2_as.append(avg_pairs2[0])
        pairs2_std_as.append(std_pairs2[0])

        ub_as.append(avg_ub1[0])
        ub_std_as.append(std_ub1[0])
        ub2_as.append(avg_ub2[0])
        ub2_std_as.append(std_ub2[0])

        worst_as.append(avg_worst1[0])
        worst_std_as.append(std_worst1[0])

        worst2_as.append(avg_worst2[0])
        worst2_std_as.append(std_worst2[0])
        
    fig, ax = plt.subplots()

    nameRefactor = {AlgorithmName.ILSR_RS: 'ilsr (repeat samples)'}

    # Define series with labels and markers
    series = [
        (f"Pair ({nameRefactor.get(algoName1, algoName1)})", pairs_as, pairs_std_as, "o"),
        (f"Pair ({nameRefactor.get(algoName2, algoName2)})", pairs2_as, pairs2_std_as, "s"),
        (f"Lower Bound ({nameRefactor.get(algoName1, algoName1)})", worst_as, worst_std_as, "o"),
        (f"Lower Bound ({nameRefactor.get(algoName2, algoName2)})", worst2_as, worst2_std_as, "s"),
        (f"Upper-Bound ({nameRefactor.get(algoName1, algoName1)})", ub_as, ub_std_as, "o"),
        (f"Upper-Bound ({nameRefactor.get(algoName2, algoName2)})", ub2_as, ub2_std_as, "s"),
    ]

    for label, avg, std, marker in series:
        print(label, avg, std)
        ax.errorbar(
            alphas,
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


    ymin = 0
    ymax = min(2.1, max(max(pairs_as) + max(pairs_std_as), 
                        max(pairs2_as) + max(pairs2_std_as), 
                        max(ub_as) + max(ub_std_as),
                        max(ub2_as) + max(ub2_std_as)) * 1.1)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("$\\alpha$")
    if d1:
        ax.set_ylabel("$\ell_1$-error")
    else:
        ax.set_ylabel("$\ell_\infty$-error")

    dsName = DatasetName.POWERLAW
    dsType = str(size)
    ax.legend(loc="best")

    fig.tight_layout()

    current_file_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'plots',f"{dsName}_{dsType}", 'multiple_alpha'))
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.abspath(os.path.join(output_dir, f'{dsName}_{dsType}_{algoName1}_vs_{algoName2}_'+('d1' if d1 else 'dinfty')))

    fig.savefig(f"{base_path}.{extension}")

    plt.close(fig)


if __name__ == '__main__':
    
    eps = 0.1
    alphas = [0.1, 0.5, 1, 2]

    for size in [250, 5000]:
        for extension in ['png', 'pdf']:
            for d1 in [True]:
                print(f'plotting size:{size}, alphas:{alphas}', d1, extension)
                plotMultipleAlphas(alphas, size, eps, d1_error=d1, extension=extension)