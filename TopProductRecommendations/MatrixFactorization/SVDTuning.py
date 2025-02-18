# -*- coding: utf-8 -*-
"""
@author: Swati
"""

from PC import PC
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

import random
import numpy as np

def LoadPCData():
    ml = PC()
    print("Loading product ratings...")
    data = ml.loadPCLatestSmall()
    print("\nComputing product popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadPCData()

print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
