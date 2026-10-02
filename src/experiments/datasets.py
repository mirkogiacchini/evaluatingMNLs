from utils.MNL import *
import os, json, csv
import numpy as np 

def readCodeforcesDataset(type:str, rng=None, maxRating=False) -> MNL:
    assert type in ['competing', 'active', 'full']

    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    pathCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'codeforces', f'{type}Users.json'))
    
    d = None
    with open(pathCF, 'r') as f:
        d = json.load(f)
    assert d is not None 

    ratings = d['maxRating' if maxRating else 'rating']
    
    log10 = np.log(10)
    weights = [r / 400 * log10 for r in ratings]
    
    return MNL(rng, weights, logweights=True,name=f"codeforces-{type}")

def readChessDataset(type:str, rng=None, maxRating:bool=False) -> MNL:
    assert type in ['bullet', 'blitz']

    current_file_dir = os.path.dirname(__file__)  
    chessPath = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'chess.com', f'{type}_ratings.csv'))

    offset = 1 if maxRating else 0

    with open(chessPath, mode = 'r') as f:
        ratings = [line[2+offset] for line in csv.reader(f)][1:]
    
    ratings = [float(x) for x in ratings]
    log10 = np.log(10)
    weights = [r / 400 * log10 for r in ratings]

    return MNL(rng, weights, logweights=True, name=f"chess.com-{type}")

def readTennisDataset(type:str, rng=None)->MNL:
    assert type in ['female_elo','female_yelo','male_elo','male_yelo']

    current_file_dir = os.path.dirname(__file__)  
    tennisPath = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'tennisabstract', f'tennis_{type}.csv'))

    with open(tennisPath, mode = 'r') as f:
        ratings = [line[1] for line in csv.reader(f)][1:]
    
    ratings = [float(x) for x in ratings]
    log10 = np.log(10)
    weights = [r / 400 * log10 for r in ratings]

    return MNL(rng, weights, logweights=True, name=f"tennis-{type}")

# ---------------- some synthetic MNLs
def evenly_spaced_indices(min_val, max_val, k):
    return np.round(
        np.linspace(min_val, max_val, k)
    ).astype(int)

def skip_items_MNL(dsName, type, num_items, rng=None):
    if dsName == 'codeforces':
        M = readCodeforcesDataset(type)
    elif dsName == 'chess.com':
        M = readChessDataset(type)
    else:
        M = readTennisDataset(type)
    
    weights = sorted(M.weights, reverse=True)
    assert len(weights) >= num_items
    weights = [weights[i] for i in evenly_spaced_indices(0, len(weights)-1, num_items)]
    assert len(weights) == num_items

    return MNL(rng, weights, logweights=M.logweights, name=f"zskip_{dsName}-{type}")
