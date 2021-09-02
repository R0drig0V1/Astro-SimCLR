import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

sys.path.append('utils')
import utils_dataset
from utils_dataset import dataset_structure

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)


#x_train = data['Train']['images']
#x_val = data['Validation']['images']
#x_test = data['Test']['images']

#y_train = data['Train']['labels']
#y_val = data['Validation']['labels']
#y_test = data['Test']['labels']