from experiments.run_experiments import DatasetName
from experiments.datasets import *

def readMaxMinRatio(dsName, dsType):
    if dsName == DatasetName.CHESSCOM:
        M1 = readChessDataset(dsType, None)
    elif dsName == DatasetName.TENNIS:
        M1 = readTennisDataset(dsType, None)
    elif dsName == DatasetName.CODEFORCES:
        M1 = readCodeforcesDataset(dsType, None)
    elif dsName == DatasetName.SKIP:
        ddsname, ddstype, num_items = dsType.split('#')
        M1 = skip_items_MNL(ddsname, ddstype, int(num_items))
    elif dsName == DatasetName.POWERLAW:
        alpha, num_items = dsType.split('#')
        M1 = powerlaw_mnl(int(num_items), alpha=float(alpha))

    M1.convertToLogWeights()
    print(f'{np.max(M1.weights)}, {np.min(M1.weights)}')
    return np.exp(np.max(M1.weights) - np.min(M1.weights))

if __name__ == '__main__':
    datasets = [(DatasetName.TENNIS, 'female_elo'), (DatasetName.TENNIS, 'male_elo'), (DatasetName.TENNIS, 'female_yelo'), (DatasetName.TENNIS, 'male_yelo'),
            (DatasetName.CHESSCOM, 'bullet'), (DatasetName.CHESSCOM, 'blitz'),
            (DatasetName.CODEFORCES, 'competing'), (DatasetName.CODEFORCES, 'active'), (DatasetName.CODEFORCES, 'full'),

            (DatasetName.SKIP, 'codeforces#active#500'), (DatasetName.SKIP, 'codeforces#competing#500'), (DatasetName.SKIP, 'codeforces#full#500'), 
            (DatasetName.SKIP, 'chess.com#bullet#500'), (DatasetName.SKIP, 'chess.com#blitz#500'),
            ]

    datasets = []
    sizes = [250, 500] + [1000 * i for i in range(1, 6)]
    alphas = [2, 1, 0.5, 0.1]
    for size in sizes:
        for alpha in alphas:
           datasets.append((DatasetName.POWERLAW, f'{alpha}#{size}'))

    for dsName, dsType in datasets:
        print(f'Max / Min in {dsName}-{dsType}: {readMaxMinRatio(dsName, dsType)}')