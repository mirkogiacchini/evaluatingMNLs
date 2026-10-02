from plotting.plot_results import stylePlot, readResults
from experiments.run_experiments import *
import matplotlib.pyplot as plt
import os
import numpy as np

def _readsinglealgo(output_dir, seeds, d1_error):
    nquer = []
    pairs_d1 = []
    worst_d1 = []
    ub_d1 = []
    fullslate_d1 = []
    fullslate_mg15 = []
    fullslate_noh12 = []
    avg_pair = []
    median_pair = []

    for seed in seeds:
        output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))

        with open(output_file, 'r') as f:
            res = json.load(f)

            nquer.append(res['num_queries'])
            if d1_error:
                pairs_d1.append((res['lower_bound_pairs_d1'] + res['upper_bound_pairs_d1']) / 2)
                worst_d1.append(res['worst_slate_d1_error'])
                ub_d1.append(res['upper_bound_d1'])
                fullslate_d1.append(res['full_slate_ell1'])
                fullslate_mg15.append(res['logs_rmse_mg15'])
                fullslate_noh12.append(res['ell2_norm_fullslate_noh12'])
            
                avg_pair.append(res['avg_pair_d1'])
                median_pair.append(res['median_pair_d1'])
            else:
                pairs_d1.append((res['lower_bound_pairs_d1'] + res['upper_bound_pairs_d1']) / 4)
                worst_d1.append(res['worst_slate_dinfty_error'])
                ub_d1.append(res['upper_bound_dinfty'])
                fullslate_d1.append(res['full_slate_ellinf'])
                fullslate_mg15.append(res['logs_rmse_mg15'])
                fullslate_noh12.append(res['ell2_norm_fullslate_noh12'])
            
                avg_pair.append(res['avg_pair_d1'] / 2)
                median_pair.append(res['median_pair_d1'] / 2)

            n = len(res['M1']['weights'])
    return n, nquer, pairs_d1, worst_d1, ub_d1, fullslate_d1, fullslate_mg15, fullslate_noh12, avg_pair, median_pair

def readResults1vs1(dsName, dsType, algoName1, algoName2, queries, seeds, d1_error = True):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located

    num_queries = []

    avg_pair_error_d1 = []
    std_pair_error_d1 = []

    avg_worst_error_d1 = []
    std_worst_error_d1 = []

    avg_d1_upper_bound = []
    std_d1_upper_bound = []

    avg_fullslate_d1 = []
    std_fullslate_d1 = []

    avg_fullslate_rmse_mg15 = []
    std_fullslate_rmse_mg15 = []

    avg_fullslate_ell2_noh12 = []
    std_fullslate_ell2_noh12 = []

    avg_avg_pair_d1 = []
    std_avg_pair_d1 = []

    avg_median_pair_d1 = []
    std_median_pair_d1 = []
    
    
    for numQueries in queries:
        output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName1), str(numQueries)))
        n, nquer, pairs_d1, worst_d1, ub_d1, fullslate_d1, fullslate_mg15, fullslate_noh12, avg_pair, median_pair = _readsinglealgo(output_dir, seeds, d1_error)

        output_dir_A2 = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName2), str(numQueries)))
        n_A2, nquer_A2, pairs_d1_A2, worst_d1_A2, ub_d1_A2, fullslate_d1_A2, fullslate_mg15_A2, fullslate_noh12_A2, avg_pair_A2, median_pair_A2 = _readsinglealgo(output_dir_A2, seeds, d1_error)

        assert n == n_A2 and nquer == nquer_A2
        num_queries.append(np.mean(nquer))

        pairs_d1 = np.array(pairs_d1) - np.array(pairs_d1_A2)
        worst_d1 = np.array(worst_d1) - np.array(worst_d1_A2)
        ub_d1 = np.array(ub_d1) - np.array(ub_d1_A2)
        fullslate_d1 = np.array(fullslate_d1) - np.array(fullslate_d1_A2)
        fullslate_mg15 = np.array(fullslate_mg15) - np.array(fullslate_mg15_A2)
        fullslate_noh12 = np.array(fullslate_noh12) - np.array(fullslate_noh12_A2)
        avg_pair = np.array(avg_pair) - np.array(avg_pair_A2)
        median_pair = np.array(median_pair) - np.array(median_pair_A2)

        avg_pair_error_d1.append(np.mean(pairs_d1))
        std_pair_error_d1.append(np.std(pairs_d1) if len(pairs_d1) > 1 else 0)

        avg_worst_error_d1.append(np.mean(worst_d1))
        std_worst_error_d1.append(np.std(worst_d1) if len(worst_d1) > 1 else 0)

        avg_d1_upper_bound.append(np.mean(ub_d1))
        std_d1_upper_bound.append(np.std(ub_d1) if len(ub_d1) > 1 else 0)

        avg_fullslate_d1.append(np.mean(fullslate_d1))
        std_fullslate_d1.append(np.std(fullslate_d1) if len(fullslate_d1) > 1 else 0)

        avg_fullslate_rmse_mg15.append(np.mean(fullslate_mg15))
        std_fullslate_rmse_mg15.append(np.std(fullslate_mg15) if len(fullslate_mg15) > 1 else 0)

        avg_fullslate_ell2_noh12.append(np.mean(fullslate_noh12))
        std_fullslate_ell2_noh12.append(np.std(fullslate_noh12) if len(fullslate_noh12) > 1 else 0)

        avg_avg_pair_d1.append(np.mean(avg_pair))
        std_avg_pair_d1.append(np.std(avg_pair) if len(avg_pair) > 1 else 0)
    
        avg_median_pair_d1.append(np.mean(median_pair))
        std_median_pair_d1.append(np.std(median_pair) if len(median_pair) > 1 else 0)

    return n, num_queries, avg_pair_error_d1, std_pair_error_d1, avg_worst_error_d1, \
                std_worst_error_d1, avg_d1_upper_bound, std_d1_upper_bound, avg_fullslate_d1, std_fullslate_d1, \
                avg_fullslate_rmse_mg15, std_fullslate_rmse_mg15, avg_fullslate_ell2_noh12, std_fullslate_ell2_noh12, \
                avg_avg_pair_d1, std_avg_pair_d1, avg_median_pair_d1, std_median_pair_d1


def plotAlgo1VsAlgo2(dsName, dsType, algoName, algoName2, queries=None, seeds=None, extension='png', d1=True, plot_difference = False):
    stylePlot()

    if queries is None:
        queries = [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
        if dsName == DatasetName.CODEFORCES and AlgorithmName.MM in [algoName, algoName2]:
            queries = [0.1, 0.05, 0.01, 0.005]
        
    if seeds is None:
        seeds = list(range(42, 42 + 10))
        if dsName == DatasetName.CODEFORCES and AlgorithmName.MM in [algoName, algoName2]:
            seeds = [42]
    
    if plot_difference:
        (n, numqueries,
        avg_pairs, std_pairs,
        avg_worst, std_worst,
        avg_ub,    std_ub,
        avg_full,  std_full,
        avg_mg15, std_mg15,
        avg_noh12, std_noh12,
        avg_avg_pair, std_avg_pair,
        avg_median_pair, std_median_pair) = readResults1vs1(dsName, dsType, algoName, algoName2, queries, seeds, d1_error=d1)
    else:
        (n, numqueries,
        avg_pairs1, std_pairs1,
        avg_worst1, std_worst1,
        avg_ub1,    std_ub1,
        avg_full1,  std_full1,
        avg_mg151, std_mg151,
        avg_noh121, std_noh121,
        avg_avg_pair1, std_avg_pair1,
        avg_median_pair1, std_median_pair1) = readResults(dsName, dsType, algoName, queries, seeds, d1_error=d1)

        (n2, numqueries2,
        avg_pairs2, std_pairs2,
        avg_worst2, std_worst2,
        avg_ub2,    std_ub2,
        avg_full2,  std_full2,
        avg_mg152, std_mg152,
        avg_noh122, std_noh122,
        avg_avg_pair2, std_avg_pair2,
        avg_median_pair2, std_median_pair2) = readResults(dsName, dsType, algoName2, queries, seeds, d1_error=d1)

        assert n == n2 and numqueries == numqueries2


    nlog2n = n * np.log(n) / np.log(2)
    numqueries = [nq / nlog2n for nq in numqueries]

    fig, ax = plt.subplots()

    nameRefactor = {AlgorithmName.ILSR_RS: 'ilsr-with-repetitions'}
    
    # Define series with labels and markers
    if plot_difference:
        series = [
            ("Pairs",                    avg_pairs, std_pairs, "o"),
            ("Lower Bound (Worst Slate)", avg_worst, std_worst, "s"),
            ("Upper Bound",              avg_ub,    std_ub,    "^"),
            ("Full-Slate",               avg_full,  std_full,  "D"),
            ("Avg Pairs",                    avg_avg_pair, std_avg_pair, "H"),
            #("Full-Slate (RMSE)",               avg_mg15,  std_mg15,  "v"),
            #("Full-Slate (normalized $\ell_2$)",   avg_noh12,  std_noh12,  "P"),
            #("Median Pairs",                    avg_median_pair, std_median_pair, "*"),
        ]
    else:
        series = [
            (f"Pairs ({nameRefactor.get(algoName, algoName)})", avg_pairs1, std_pairs1, "o"),
            (f"Pairs ({nameRefactor.get(algoName2, algoName2)})", avg_pairs2, std_pairs2, "s"),
            (f"Full-Slate ({nameRefactor.get(algoName, algoName)})", avg_full1, std_full1, "^"),
            (f"Full-Slate ({nameRefactor.get(algoName2, algoName2)})", avg_full2, std_full2, "D"),
        ]

    for label, avg, std, marker in series:
        ax.errorbar(
            numqueries,
            avg,
            yerr=std,
            label=label,
            marker=marker,
            #linestyle="-",
            capsize=2,
            elinewidth=.8,
            linewidth=1, #1.5
        )

    ax.set_xscale("log")

    if plot_difference:
        ymin = max(-2, min(min(avg_pairs), min(avg_worst), min(avg_ub), min(avg_full), min(avg_avg_pair)) - max(max(std_pairs), max(std_worst), max(avg_ub), max(std_full), max(std_avg_pair))*1.1)
        ymax = min(2.1, (max(avg_ub) + max(std_ub)) * 1.1)
    else:
        ymin = 0
        ymax = min(2.1, max(max(avg_pairs1) + max(std_pairs1), max(avg_pairs2) + max(std_pairs2)) * 1.1)

    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Number of Queries / $n \log_2 n$")
    if d1:
        if plot_difference:
            ax.set_ylabel(f"$\ell_1$ difference: {nameRefactor.get(algoName, algoName)} $-$ {nameRefactor.get(algoName2, algoName2)}")
        else:
            ax.set_ylabel("$\ell_1$-error")
    else:
        if plot_difference:
            ax.set_ylabel(f"$\ell_\infty$ difference: {nameRefactor.get(algoName, algoName)} $-$ {nameRefactor.get(algoName2, algoName2)}")
        else:
            ax.set_ylabel("$\ell_\infty$-error")
    type_translator = {'female_elo': 'women', 'male_elo':'men', 'female_yelo': 'women-2025', 'male_yelo': 'men-2025'}
    ax.set_title(f"Dataset: {dsName}-{type_translator.get(dsType, dsType)} ($n={n}$)")
    ax.legend(loc="best")

    fig.tight_layout()

    current_file_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(
        os.path.join(current_file_dir, '..', '..', 'plots',
                     f"{dsName}_{dsType}", f"comparison")
    )
    os.makedirs(output_dir, exist_ok=True)

    base_path = os.path.abspath(os.path.join(output_dir, f'{dsName}_{dsType}_{algoName}_vs_{algoName2}_'+('d1' if d1 else 'dinfty')))

    fig.savefig(f"{base_path}.{extension}")

    plt.close(fig)

def main(extension):
    datasets = [
                (DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full')
            ]
    
    for d1 in [True, False]:
        for (dsname, dstype) in datasets:
            print(dsname, dstype, d1)
            if dsname != DatasetName.CODEFORCES:
                plotAlgo1VsAlgo2(dsname, dstype, AlgorithmName.ILSR, AlgorithmName.LOBSTER, d1=d1, extension=extension, plot_difference=False)

                plotAlgo1VsAlgo2(dsname, dstype, AlgorithmName.ILSR, AlgorithmName.ILSR_RS, d1=d1, extension=extension, plot_difference=False)
            else:                
                plotAlgo1VsAlgo2(dsname, dstype, AlgorithmName.MM, AlgorithmName.LOBSTER, d1=d1, extension=extension, plot_difference=False)

if __name__ == '__main__':
    for extension in ['png', 'pdf']:
        main(extension)