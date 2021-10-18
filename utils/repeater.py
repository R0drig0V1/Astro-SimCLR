import numpy as np
import pandas as pd
import os
import re 


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
def folder_best_hyperparameters(df):

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

		# Save folder of the best checkpoint
		best_folder = ''

		# Save epoch of the best checkpoint (later checkpoint is better)
		best_epoch = 0

		# Iterate the folders inside the experiment
		for folder in os.listdir(exp_folder):

			# If the folder contents a checkpoint
			if 'checkpoint_' in folder:

				# From the name extract the epoch
				epoch = int(re.search(r'epoch=(.*?)-', folder).group(1))

				# Update epoch and folder
				if (epoch >= best_epoch) :
					best_epoch = epoch
					best_folder = folder

		# Save checkpoint folder
		checkpoint_path = os.path.join(exp_folder, best_folder, "checkpoint.ckpt")
		checkpoint_paths.append(checkpoint_path)

	return checkpoint_paths

