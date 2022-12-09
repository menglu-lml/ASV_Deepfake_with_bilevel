import sys
import numpy as np
import pandas as pd

args = sys.argv

model_1 = args[1]
model_2 = args[2]

score_1 = pd.read_csv(model_1, sep=" ", header=None)
score_2 = pd.read_csv(model_2, sep=" ", header=None)

score_ensemble = score_2
score_ensemble.iloc[:,3] = (score_1[3]*0.6+ score_2[3]*0.4)
score_ensemble.to_csv('Test_Score_File/Ensemble.txt', header=None, index=None, sep=' ')