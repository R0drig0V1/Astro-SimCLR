import numpy as np
import pandas as pd
import os
import re 

# -----------------------------------------------------------------------------

# Return names of hyperparameter columns of dataframe
def hyperparameter_columns(df):

	# Save columns with hyperparameters
	hyper_columns = []

	# Iterate dataframe columns
	for column in df.columns:

		# Save column name
		if (('config' in column) & (not ('__trial_index__' in column))):
			hyper_columns.append(column)

	return hyper_columns


# Create summary of hyperparamter combinations (compute mean and std)
def summary(df):

	# Names of hyperparameter columns of dataframe
	hyper_columns = hyperparameter_columns(df)

	# Compute mean and std of accuracies
	df_summary = df.groupby(hyper_columns)["accuracy"].agg([np.mean, np.std]).reset_index()

	return df_summary


# Return the list of the checkpoints of the best hyperparamter combination
def path_best_hyperparameters(df):

	# Names of hyperparameter columns of dataframe
	hyper_columns = hyperparameter_columns(df)

	# Summary of hyperparamter combinations 
	df_summary = summary(df)

	# Index of best combination of hyperparamters
	index = df_summary['mean'].argmax()

	# Best combination of hyperparameters		
	best_config = list(df_summary.loc[index])

	# Mask of the best hyperparamter combination
	where = 1
	for hyper_name, opt_hyper in zip(hyper_columns, best_config):
		where = where & (df[hyper_name] == opt_hyper)

	# Folders of the best hyperparameter combination
	folders = list(df['logdir'][where])

	# Save checkpoint paths
	checkpoint_paths = []

	# For each realization of the hyperparameter combination
	for exp_folder in folders:

		# Save checkpoint folder
		checkpoint_path = os.path.join(exp_folder, "checkpoint.ckpt")
		checkpoint_paths.append(checkpoint_path)

	return checkpoint_paths

# -----------------------------------------------------------------------------
