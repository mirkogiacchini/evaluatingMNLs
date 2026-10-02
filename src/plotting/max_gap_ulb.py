import os, json, tqdm
from plotting.plot_results import readResults
from experiments.run_experiments import AlgorithmName, DatasetName

def gap_ub_lb(dsName, dstype, algoName, d1, use_pairs=True):
    queries = [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    if dsName == DatasetName.CODEFORCES and algoName == AlgorithmName.MM:
        queries = [0.1, 0.05, 0.01, 0.005]
    
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
     avg_median_pair, std_median_pair) = readResults(dsName, dstype, algoName, queries, seeds, d1_error=d1)

    if use_pairs:
        ratios = [ub / max(pair, worst) for pair, worst, ub in zip(avg_pairs, avg_worst, avg_ub)]
    else:
        ratios = [ub / full for full, ub in zip(avg_full, avg_ub)]
    return max(ratios)

if __name__ == "__main__":
    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full')
            ]
    algorithms = [AlgorithmName.ILSR, AlgorithmName.LOBSTER, AlgorithmName.MM]

    for use_pair in [True, False]:
        print('Gap With PAIRS' if use_pair else 'Gap with FULL-SLATE')
        for d1 in [True, False]:
            for (dsname, dstype) in datasets:
                for algo in algorithms:
                    if dsname == DatasetName.CODEFORCES and algo == AlgorithmName.ILSR:
                        continue
                    if dsname == DatasetName.CHESSCOM and algo == AlgorithmName.MM:#TO BE REMOVED!!!
                        continue 
                    if dstype == 'full': #TO BE REMOVED!!!
                        continue
                    
                    gap = gap_ub_lb(dsname, dstype, algo, d1, use_pairs=use_pair)
                    print(f'max gap UB/LB in dataset: {dsname}-{dstype}; algo: {algo} (d1={d1})  =  {gap}')

        print('\n\n')