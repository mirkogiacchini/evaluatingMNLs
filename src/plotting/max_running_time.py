import os, json, tqdm
from plotting.plot_results import readResults
import numpy as np
from experiments.run_experiments import AlgorithmName, DatasetName

def readRunningTimes(dsName, dsType, algoName, queries, seeds):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located

    avg_pairs = []
    std_pairs = []
    avg_worst1 = []
    std_worst1 = []
    avg_worstInf = []
    std_worstInf = []
    avg_ub1 = []
    std_ub1 = []
    avg_ubInf = []
    std_ubInf = []
    
    for numQueries in queries:
        output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName), str(numQueries)))
        
        pairs = []
        worst_1 = []
        worst_inf = []
        ub_1 = []
        ub_inf = []

        for seed in seeds:
            output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))

            with open(output_file, 'r') as f:
                res = json.load(f)

                pairs.append(res['time_s_worst_pair'])
                worst_1.append(res['worst_slate_d1_time_s'])
                worst_inf.append(res['worst_slate_dinfty_time_s'])
                ub_1.append(res['upper_bound_d1_time_s'])
                ub_inf.append(res['upper_bound_dinfty_time_s'])                

                n = len(res['M1']['weights'])
            
        avg_pairs.append(np.mean(pairs))
        std_pairs.append(np.std(pairs) if len(pairs) > 1 else 0)

        avg_worst1.append(np.mean(worst_1))
        std_worst1.append(np.std(worst_1) if len(worst_1) > 1 else 0)

        avg_worstInf.append(np.mean(worst_inf))
        std_worstInf.append(np.std(worst_inf) if len(worst_inf) > 1 else 0)

        avg_ub1.append(np.mean(ub_1))
        std_ub1.append(np.std(ub_1) if len(ub_1) > 1 else 0)

        avg_ubInf.append(np.mean(ub_inf))
        std_ubInf.append(np.std(ub_inf) if len(ub_inf) > 1 else 0)

    return n, avg_pairs, std_pairs, avg_worst1, std_worst1, avg_worstInf, std_worstInf, avg_ub1, std_ub1, avg_ubInf, std_ubInf

def worstEntry(mean, std):
    m, idx = None, None
    for i in range(len(mean)):
        v = mean[i]
        if m is None or v > m:
            m = v
            idx = i 
    return mean[idx], std[idx]

def worst_running_time(dsName, dstype, algoName):
    queries = [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    if dsName == DatasetName.CODEFORCES and algoName == AlgorithmName.MM:
        queries = [0.1, 0.05, 0.01, 0.005]
    
    seeds = list(range(42, 42 + 10))
    if dsName == DatasetName.CODEFORCES and algoName == AlgorithmName.MM:
        seeds = [42]

    (n, 
     avg_pairs, std_pairs, 
     avg_worst1, std_worst1, 
     avg_worstInf, std_worstInf, 
     avg_ub1, std_ub1, 
     avg_ubInf, std_ubInf) = readRunningTimes(dsName, dstype, algoName, queries, seeds)
 
    return worstEntry(avg_pairs, std_pairs), \
            worstEntry(avg_worst1, std_worst1), \
            worstEntry(avg_worstInf, std_worstInf),\
            worstEntry(avg_ub1, std_ub1), \
            worstEntry(avg_ubInf, std_ubInf),

if __name__ == "__main__":
    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full')
            ]
    algorithms = [AlgorithmName.ILSR, AlgorithmName.LOBSTER, AlgorithmName.MM]

    for (dsname, dstype) in datasets:
        for algo in algorithms:
            if dsname == DatasetName.CODEFORCES and algo == AlgorithmName.ILSR:
                continue
            if dsname == DatasetName.CHESSCOM and algo == AlgorithmName.MM:#TO BE REMOVED!!!
                continue 
            if dstype == 'full' and algo == AlgorithmName.MM: #TO BE REMOVED!!!
                continue
            
            pairs, worst1, worstInf, ub1, ubInf = worst_running_time(dsname, dstype, algo)
            print(f'max gap UB/LB in dataset: {dsname}-{dstype}; algo: {algo}  =  Pairs: {pairs[0]}±{pairs[1]}, Worst1: {worst1[0]}±{worst1[1]}, UB1:{ub1[0]}±{ub1[1]}')