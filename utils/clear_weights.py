import os
import shutil

import pandas as pd

# -----------------------------------------------------------------------------

# Dataframe with results
results_ce = pd.read_csv('../results/hyperparameter_tuning_ce.csv', index_col=0)
results_simclr = pd.read_csv('../results/hyperparameter_tuning_simclr.csv', index_col=0)

def del_folder(results, path_exp):

	exp = os.listdir(path_exp)
	logdirs = list(results['logdir'].apply(lambda dir: dir[dir.find('/id')+1:]))

	for folder in exp:
		if ((not (folder in logdirs)) and ('id' in folder)):

			path = path_exp + folder

			if os.path.exists(path):
				print("dir: ", path)
				shutil.rmtree(path)
			else:
				print("File not found in the directory")

del_folder(results_ce, '../../hyperparameter_tuning/CE/')
del_folder(results_simclr, '../../hyperparameter_tuning/SimCLR/')
