from experiments.run_experiments import *
import json
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def readResults(dsName, dsType, algoName, queries, seeds, d1_error = True, include_log_weights_dist=False):
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
    
    avg_ell1_log_weights = []
    std_ell1_log_weights = []
    
    avg_ell_inf_log_weights = []
    std_ell_inf_log_weights = []
    
    for numQueries in queries:
        output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName), str(numQueries)))
        
        nquer = []
        pairs_d1 = []
        worst_d1 = []
        ub_d1 = []
        fullslate_d1 = []
        fullslate_mg15 = []
        fullslate_noh12 = []
        avg_pair = []
        median_pair = []

        ell1_log_weights = []
        ell_inf_log_weights = []

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
                    ub_d1.append(min(res['upper_bound_dinfty'], res['upper_bound_d1'] / 2))
                    fullslate_d1.append(res['full_slate_ellinf'])
                    fullslate_mg15.append(res['logs_rmse_mg15'])
                    fullslate_noh12.append(res['ell2_norm_fullslate_noh12'])
                
                    avg_pair.append(res['avg_pair_d1'] / 2)
                    median_pair.append(res['median_pair_d1'] / 2)

                M1 = MNL.fromDict(res['M1'])
                M2 = MNL.fromDict(res['M2'])
                M1.convertToLogWeights()
                M2.convertToLogWeights
                ell1_log_weights.append(np.linalg.norm(np.array(M1.weights) - np.array(M2.weights), ord=1))
                ell_inf_log_weights.append(np.linalg.norm(np.array(M1.weights) - np.array(M2.weights), ord=np.inf))

                n = len(res['M1']['weights'])
            

        num_queries.append(np.mean(nquer))

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

        avg_ell1_log_weights.append(np.mean(ell1_log_weights))
        std_ell1_log_weights.append(np.std(ell1_log_weights))
        
        avg_ell_inf_log_weights.append(np.mean(ell_inf_log_weights))
        std_ell_inf_log_weights.append(np.std(ell_inf_log_weights))
    
    if include_log_weights_dist:
        return n, num_queries, avg_pair_error_d1, std_pair_error_d1, avg_worst_error_d1, \
                    std_worst_error_d1, avg_d1_upper_bound, std_d1_upper_bound, avg_fullslate_d1, std_fullslate_d1, \
                    avg_fullslate_rmse_mg15, std_fullslate_rmse_mg15, avg_fullslate_ell2_noh12, std_fullslate_ell2_noh12, \
                    avg_avg_pair_d1, std_avg_pair_d1, avg_median_pair_d1, std_median_pair_d1, \
                    avg_ell1_log_weights, std_ell1_log_weights, avg_ell_inf_log_weights, std_ell_inf_log_weights
        
    return n, num_queries, avg_pair_error_d1, std_pair_error_d1, avg_worst_error_d1, \
                std_worst_error_d1, avg_d1_upper_bound, std_d1_upper_bound, avg_fullslate_d1, std_fullslate_d1, \
                avg_fullslate_rmse_mg15, std_fullslate_rmse_mg15, avg_fullslate_ell2_noh12, std_fullslate_ell2_noh12, \
                avg_avg_pair_d1, std_avg_pair_d1, avg_median_pair_d1, std_median_pair_d1

def readExactError(dsName, dsType, algoName, queries, seeds):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located

    num_queries = []

    avg_exact_d1 = []
    std_exact_d1 = []
    
    for numQueries in queries:
        output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName), str(numQueries)))
        
        nquer = []
        exact_d1 = []

        for seed in seeds:
            output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))

            with open(output_file, 'r') as f:
                res = json.load(f)
                nquer.append(res['num_queries'])
                exact_d1.append(res['worst_d1_exact'])
                n = len(res['M1']['weights'])
            

        num_queries.append(np.mean(nquer))
        avg_exact_d1.append(np.mean(exact_d1))
        std_exact_d1.append(np.std(exact_d1))
        
    return n, num_queries, avg_exact_d1, std_exact_d1


def stylePlot():
    # Colorblind-friendly Okabe–Ito palette
    color_cycle = ['#0072B2',  # blue
                   '#D55E00',  # vermillion
                   '#009E73',  # green
                   '#CC79A7',  # reddish purple
                   '#F0E442',  # yellow
                   '#56B4E9',  # sky blue  
                   '#E69F00',  # coral
                   '#0099A8']  # teal  

    plt.rcParams.update({
        # --- Figure / save ---
        "figure.dpi": 300,
        "savefig.dpi": 600,          # high-res PNG
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.figsize": (4.5, 3.2),   # good for 2-column width when scaled

        # --- Fonts / text ---
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # --- Lines / markers ---
        "lines.linewidth": 1, #1.5
        "lines.markersize": 3, #5

        # --- Axes style ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,

        # --- Legend ---
        "legend.frameon": False,

        # --- Tick direction ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    linestyles = ['-', '--']

    # Combine so every line gets a unique style/color pairing
    plt.rcParams['axes.prop_cycle'] = (
        cycler(color=color_cycle) +
        cycler(linestyle=linestyles * (len(color_cycle) // len(linestyles)))
    )

    # Apply color cycle
    #plt.rcParams["axes.prop_cycle"] = cycler(color=color_cycle)


def plotResult(dsName, dsType, algoName, queries=None, seeds=None, extension='png', d1=True, full_results=False):
    stylePlot()

    if queries is None:
        queries = [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
        if dsName == DatasetName.CODEFORCES and algoName == AlgorithmName.MM:
            queries = [0.1, 0.05, 0.01, 0.005]
        
    if seeds is None:
        seeds = list(range(42, 42 + 10))
        if dsName == DatasetName.CODEFORCES and algoName == AlgorithmName.MM:
            seeds = [42]


    (n, numqueries,
     avg_pairs, std_pairs,
     avg_worst, std_worst,
     avg_ub,    std_ub,
     avg_full,  std_full,
     avg_mg15, std_mg15,
     avg_noh12, std_noh12,
     avg_avg_pair, std_avg_pair,
     avg_median_pair, std_median_pair,
     avg_ell1_log_weight, std_ell1_log_weight,
     avg_ell_inf_log_weight, std_ell_inf_log_weight) = readResults(dsName, dsType, algoName, queries, seeds, d1_error=d1, include_log_weights_dist=True)

    nlog2n = n * np.log(n) / np.log(2)
    numqueries = [nq / nlog2n for nq in numqueries]

    fig, ax = plt.subplots()

    # Define series with labels and markers
    series = [
        ("Pairs",                    avg_pairs, std_pairs, "o"),
        ("Lower Bound (Worst Slate)", avg_worst, std_worst, "s"),
        ("Upper Bound",              avg_ub,    std_ub,    "^"),
        ("Full-Slate",               avg_full,  std_full,  "D"),
        ("Avg Pairs",                    avg_avg_pair, std_avg_pair, "H"),
        #("Median Pairs",                    avg_median_pair, std_median_pair, "*"),
        #("$\ell_1$-log-weights",      avg_ell1_log_weight, std_ell1_log_weight, "v"),
        #("$\ell_\infty$-log-weights",      avg_ell_inf_log_weight, std_ell_inf_log_weight, "P"),
    ]

    if n <= 20:
        _, _, avg_exact_d1_err, std_exact_d1_err = readExactError(dsName, dsType, algoName, queries=queries, seeds=seeds)
        print(dsName, dsType, algoName, f"Exact-D1-error: {avg_exact_d1_err}")

    if full_results:
        series.extend([
            ("Full-Slate (RMSE)",               avg_mg15,  std_mg15,  "v"),
            ("Full-Slate (normalized $\ell_2$)",   avg_noh12,  std_noh12,  "P"),
        ])

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

    ymin = 0
    ymax = min(2.1 if d1 else 1.1, (max(avg_ub) + max(std_ub)) * 1.1)

    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Number of Queries / $n \log_2 n$")
    if d1:
        ax.set_ylabel(r"$\ell_1$-Error")
    else:
        ax.set_ylabel(r"$\ell_\infty$-Error")

    type_translator = {'female_elo': 'women', 'male_elo':'men', 'female_yelo': 'women-2025', 'male_yelo': 'men-2025'}
    nameRefactor = {AlgorithmName.ILSR_RS: 'ilsr(repeated sampling)'}

    ax.set_title(f"Dataset: {dsName}-{type_translator.get(dsType, dsType)} ($n={n}$)\nAlgorithm: {nameRefactor.get(algoName, algoName)}")
    ax.legend(loc="best")

    fig.tight_layout()

    current_file_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(
        os.path.join(current_file_dir, '..', '..', 'plots',
                     f"{dsName}_{dsType}", str(algoName))
    )
    os.makedirs(output_dir, exist_ok=True)

    file_name = f'{dsName}_{dsType}_{str(algoName)}_{"d1" if d1 else "dinfty"}'
    base_path = os.path.abspath(os.path.join(output_dir, file_name))

    fig.savefig(f"{base_path}.{extension}")

    plt.close(fig)

def main(extension):
    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full'),
                (DatasetName.POWERLAW, '1#5000')
            ]
    algorithms = [AlgorithmName.ILSR, AlgorithmName.LOBSTER, AlgorithmName.MM, 
                  AlgorithmName.ILSR_ALPHA000001, AlgorithmName.ILSR_ALPHA0001, AlgorithmName.ILSR_ALPHA001, AlgorithmName.ILSR_ALPHA01, AlgorithmName.ILSR_ALPHA1]

    for d1 in [True, False]:
        for (dsname, dstype) in datasets:
            for algo in algorithms:
                if dsname == DatasetName.CODEFORCES and algo == AlgorithmName.ILSR:
                    continue
                if dsname == DatasetName.CHESSCOM and algo == AlgorithmName.MM:
                    continue 
                if dsname == DatasetName.POWERLAW and algo == AlgorithmName.MM:
                    continue 
                if algo in [AlgorithmName.ILSR_ALPHA000001, AlgorithmName.ILSR_ALPHA0001, AlgorithmName.ILSR_ALPHA001, AlgorithmName.ILSR_ALPHA01, AlgorithmName.ILSR_ALPHA1]:
                    if dsname != DatasetName.TENNIS:
                        continue 
                print(f'plotting {dsname}-{dstype}, {algo}, d1={d1}')
                plotResult(dsname, dstype, algo, extension=extension, d1=d1, full_results=True)

if __name__ == '__main__':
    for extension in ['png', 'pdf']:
        main(extension)