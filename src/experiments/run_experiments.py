from experiments.datasets import *
from non_adaptive_algorithm.ilsr import *
from non_adaptive_algorithm.mm import *
from adaptive_algorithm.lobster_forest import *
import time, os, json
from evaluate.heuristic_worst_slates import *
from evaluate.worst_pair import *
from evaluate.upper_bounds import *
from evaluate.average import *
from evaluate.evaluate import getSlateError
import platform, math

from enum import Enum

class StrEnum39(str, Enum):
    def __str__(self):
        return self.value

class AlgorithmName(StrEnum39):
    ILSR = "ilsr"
    MM = "mm"
    LOBSTER = "lobster"
    ILSR_RS = "ilsr_rs"
    ILSR_ALPHA000001 = "ilsr_alpha1e-6"
    ILSR_ALPHA0001 = "ilsr_alpha1e-4"
    ILSR_ALPHA001 = "ilsr_alpha1e-3"
    ILSR_ALPHA01 = "ilsr_alpha1e-2"
    ILSR_ALPHA1 = "ilsr_alpha1e-1"

class DatasetName(StrEnum39):
    CODEFORCES = "codeforces"
    CHESSCOM = "chess.com"
    TENNIS = "tennis"
    SKIP = "_skip"
    POWERLAW = "_powerlaw"

def readNumberOfQueries(dsName, dsType, seed, eps):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(AlgorithmName.LOBSTER), str(eps)))
    output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))
    with open(output_file, 'r') as f:
        res = json.load(f)
        return res['num_queries']

def readLobsterMNL(dsName, dsType, seed, eps):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(AlgorithmName.LOBSTER), str(eps)))
    output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))
    with open(output_file, 'r') as f:
        res = json.load(f)
        return MNL.fromDict(res['M2']) 

def readMMPreviousMNL(dsName, dsType, seed, eps):
    all_eps = list(reversed([1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]))
    idx = all_eps.index(eps)
    if idx == 0:
        return None
    eps = all_eps[idx-1]

    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(AlgorithmName.MM), str(eps)))
    output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))
    with open(output_file, 'r') as f:
        res = json.load(f)
        return MNL.fromDict(res['M2']) 


def runAlgorithm(seed, dsName, dsType, algoName, epsilon_algo, save_result=True, run_if_duplicate=True):
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'output', str(dsName)+'_'+str(dsType), str(algoName), str(epsilon_algo)))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.abspath(os.path.join(output_dir, str(seed)+'.json'))
    if not run_if_duplicate and os.path.exists(output_file):
        return "Already run"
    
    seedSeq = np.random.SeedSequence(seed) 
    seedSeqString = seedSeq.generate_state(6)

    rng_mnl = np.random.default_rng(seedSeqString[0])
    rng_heuristics = np.random.default_rng(seedSeqString[1])
    rng_lobster = np.random.default_rng(seedSeqString[2])
    rng_avg = np.random.default_rng(seedSeqString[3])
    seed_sgd = seedSeqString[4]
    rng_perturbation = np.random.default_rng(seedSeqString[5])
    

    # Read original MNL
    if dsName == DatasetName.CHESSCOM:
        M1 = readChessDataset(dsType, rng_mnl)
    elif dsName == DatasetName.TENNIS:
        M1 = readTennisDataset(dsType, rng_mnl)
    elif dsName == DatasetName.CODEFORCES:
        M1 = readCodeforcesDataset(dsType, rng_mnl)
    elif dsName == DatasetName.SKIP:
        ddsname, ddstype, num_items = dsType.split('#')
        M1 = skip_items_MNL(ddsname, ddstype, int(num_items))
    elif dsName == DatasetName.POWERLAW:
        alpha, num_items = dsType.split('#')
        M1 = powerlaw_mnl(int(num_items), alpha=float(alpha))
    else:
        raise Exception(f"Dataset not known: {dsName}")
    
    on_first_process = True
    converged = True

    # Learn New MNL
    n = M1.getNumberOfItems()   
    if algoName == AlgorithmName.LOBSTER:
        # adaptive algorithm
        algo_time_s = time.perf_counter()
        M2, _, _ = lobsterForestAlgorithm(M1, rng_lobster, epsilon_algo, 0.01)
        algo_time_s = time.perf_counter() - algo_time_s
        num_queries = M1.totalQueriesMade()
    else:
        # non-adaptive algorithm
        num_queries = readNumberOfQueries(dsName, dsType, seed, epsilon_algo)

        if dsName == DatasetName.CODEFORCES or algoName == AlgorithmName.ILSR_RS:
            num_slates = min(int(math.ceil(10 * n * np.log(n))), num_queries)
            maxsample_dataset = M1.generateSparsePairwiseDataset(num_slates, num_queries)
        else:
            maxsample_dataset = M1.generatePairwiseDataset(num_queries, 1)

        algo_time_s = time.perf_counter()
        if algoName == AlgorithmName.ILSR or algoName == AlgorithmName.ILSR_RS:
            M2 = ILSR(n, maxsample_dataset)
        elif algoName in [AlgorithmName.ILSR_ALPHA000001, AlgorithmName.ILSR_ALPHA0001, AlgorithmName.ILSR_ALPHA001, AlgorithmName.ILSR_ALPHA01, AlgorithmName.ILSR_ALPHA1]:
            alpha = 1e-6
            if algoName == AlgorithmName.ILSR_ALPHA0001:
                alpha = 1e-4
            elif algoName == AlgorithmName.ILSR_ALPHA001:
                alpha = 1e-3
            elif algoName == AlgorithmName.ILSR_ALPHA1:
                alpha = 0.1
            elif algoName == AlgorithmName.ILSR_ALPHA01:
                alpha = 0.01
            M2 = ILSR(n, maxsample_dataset, alpha=alpha)
        elif algoName == AlgorithmName.MM:
            #start from an already trained MNL to converge faster (of course, this is not implementable in real-world, but we need this to scale to Codeforces dataset)
            initial_params = readMMPreviousMNL(dsName, dsType, seed, epsilon_algo) 
            if initial_params is None:
                initial_params = np.array(M1.weights, dtype=np.float32) # we use the real weights as initial parameters to converge faster
            else:
                initial_params.convertToLogWeights()
                initial_params = initial_params.weights
            tol=1e-9
            if dsName == DatasetName.CODEFORCES and dsType in ["active", "full"]:
                tol=1e-10
            M2, converged = mmAlgorithm(n, maxsample_dataset, initial_params=initial_params, tol=tol)
        else:
            raise Exception(f"Algo not known: {algoName}")
        algo_time_s = time.perf_counter() - algo_time_s

    if on_first_process:
        # Find worst pair  
        pair_time_s = time.perf_counter()
        worst_pair, lb_pair_error, ub_pair_error = worstPairApproximate(M1, M2, 0.01, max_iterations_in_pruned_search_timesn=8)
        pair_time_s = time.perf_counter() - pair_time_s
        
        # Find worst slates
        worst_d1_time_s = time.perf_counter()
        worst_d1_slate, worst_d1_error = heuristic_worst_slate(M1, M2, d1=True)
        worst_d1_time_s = time.perf_counter() - worst_d1_time_s

        worst_dinf_time_s = time.perf_counter()
        worst_dinf_slate, worst_dinf_error = heuristic_worst_slate(M1, M2, d1=False)
        worst_dinf_time_s = time.perf_counter() - worst_dinf_time_s

        # Find Upper Bounds
        upper_bound_d1_time_s = time.perf_counter()
        upper_bound_d1 = error_upperbound(M1, M2, d1=True)
        upper_bound_d1_time_s = time.perf_counter() - upper_bound_d1_time_s
        
        upper_bound_dinf_time_s = time.perf_counter()
        upper_bound_dinf = error_upperbound(M1, M2, d1=False)
        upper_bound_dinf_time_s = time.perf_counter() - upper_bound_dinf_time_s
        
        # errors on full slate
        full_slate_d1 = getSlateError(M1, M2, list(range(n)))
        full_slate_dinf = getSlateError(M1, M2, list(range(n)), norm=np.inf)

        erms_mg15 = mg15_ERMS(M1, M2)
        ell2norm_noh12 = noh12_normalizedNorm(M1, M2)

        # avg error
        avg_time_s = time.perf_counter()
        avg_pair_err, median_pair_err = avg_median_error_pairs_d1(M1, M2, rng_avg)
        avg_time_s = time.perf_counter() - avg_time_s

        # exact error only for very small datasets (it's exponential)
        worst_d1_exact, worst_d1_exact_slate, worst_d1_exact_s = -1, (0,1), -1
        if n <= 20:
            worst_d1_exact_s = time.perf_counter()
            worst_d1_exact_slate, worst_d1_exact = computeWorstErrorExactly(M1, M2, d1=True)
            worst_d1_exact_s = time.perf_counter() - worst_d1_exact_s

        python_environment = f'{platform.python_implementation()} {platform.python_version()}'
        machine_environment = f'{platform.system()} | {platform.release()} | {platform.machine()}'

        results = {
            'seed': seed,
            'dataset': str(dsName) + '_' + str(dsType),
            'algorithm': str(algoName),
            'num_queries': num_queries,
            'algo_time_s': str(algo_time_s),

            'full_slate_ell1': full_slate_d1,
            'full_slate_ellinf': full_slate_dinf,
            'logs_rmse_mg15': erms_mg15,
            'ell2_norm_fullslate_noh12': ell2norm_noh12,

            'worst_pair': worst_pair,
            'lower_bound_pairs_d1': lb_pair_error,
            'upper_bound_pairs_d1': ub_pair_error,
            'time_s_worst_pair': pair_time_s,

            'upper_bound_d1': upper_bound_d1,
            'upper_bound_d1_time_s': upper_bound_d1_time_s,

            'upper_bound_dinfty': upper_bound_dinf,
            'upper_bound_dinfty_time_s': upper_bound_dinf_time_s,

            'worst_slate_d1': worst_d1_slate,
            'worst_slate_d1_error': worst_d1_error,
            'worst_slate_d1_time_s': worst_d1_time_s,
            
            'worst_slate_dinfty': worst_dinf_slate,
            'worst_slate_dinfty_error': worst_dinf_error,
            'worst_slate_dinfty_time_s': worst_dinf_time_s,

            'avg_pair_d1': avg_pair_err,
            'median_pair_d1': median_pair_err,
            'avg_median_pair_time_s': avg_time_s,

            'python_environment': python_environment,
            'machine_environment': machine_environment,

            'converged': converged,

            'M1': M1.toDict(),
            'M2': M2.toDict(),
        }

        if n <= 20:
            assert worst_d1_exact >= 0, worst_d1_exact_s >= 0
            results['worst_d1_exact'] = worst_d1_exact
            results['worst_d1_exact_slate'] = worst_d1_exact_slate
            results['worst_d1_exact_time_s'] = worst_d1_exact_s

        if save_result:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        return results


if __name__ == '__main__':
    
    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
                (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
                (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full'),
                ]
    
    # add skips
    step_size = 250
    max_size = 5000
    for dsname in ['chess.com#bullet',
                   'chess.com#blitz',
                   'codeforces#competing',
                   'codeforces#active',
                   'codeforces#full',
                   ]:
        for size in range(step_size, max_size+1, step_size):
            datasets.append( (DatasetName.SKIP, f'{dsname}#{size}') )
    
    sizes = [250, 500] + [1000 * i for i in range(1, 6)]
    alphas = [2, 1, 0.5, 0.1]
    for size in sizes:
        for alpha in alphas:
           datasets.append((DatasetName.POWERLAW, f'{alpha}#{size}'))
    datasets.append((DatasetName.POWERLAW, '1#20'))
    
    eps_values = list(reversed([1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]))
    algorithms = [AlgorithmName.LOBSTER, AlgorithmName.ILSR, AlgorithmName.MM, AlgorithmName.ILSR_RS] 
    algorithms.extend([AlgorithmName.ILSR_ALPHA000001, AlgorithmName.ILSR_ALPHA0001, AlgorithmName.ILSR_ALPHA001, AlgorithmName.ILSR_ALPHA01, AlgorithmName.ILSR_ALPHA1])

    seeds = list(range(42, 42+10))

    for dsName, dsType in datasets:
        for algo in algorithms:

            for seed_idx, seed in enumerate(seeds):
                for eps in eps_values:
                    if dsName == DatasetName.CODEFORCES and algo in [AlgorithmName.ILSR, AlgorithmName.ILSR_RS]:
                        continue
                    if dsName == DatasetName.CODEFORCES and algo == AlgorithmName.MM and (eps > 0.1 or seed_idx > 0):
                        continue 
                    if dsName == DatasetName.CHESSCOM and algo == AlgorithmName.MM:
                        continue 
                    if dsName == DatasetName.SKIP and (algo == AlgorithmName.MM or algo == AlgorithmName.ILSR_RS or eps != 0.1):
                        continue 
                    if dsName == DatasetName.POWERLAW:
                        if (algo == AlgorithmName.MM or algo == AlgorithmName.ILSR_RS):
                            continue
                    if algo in [AlgorithmName.ILSR_ALPHA000001, AlgorithmName.ILSR_ALPHA0001, AlgorithmName.ILSR_ALPHA001, AlgorithmName.ILSR_ALPHA01, AlgorithmName.ILSR_ALPHA1]:
                        if dsName != DatasetName.TENNIS:
                            continue 

                    print(f'Running {algo} on {dsName} ({dsType}) with eps:{eps} -- seed: {seed}')
                    runAlgorithm(seed, dsName, dsType, algo, eps, run_if_duplicate=False)
