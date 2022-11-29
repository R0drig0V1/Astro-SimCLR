import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.dataset import repair_non_squared_stamp, preprocess_stamps

# -----------------------------------------------------------------------------

with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

features = ['sgscore1', 'distpsnr1', 'sgscore2', 'distpsnr2', 'sgscore3',
            'distpsnr3', 'isdiffpos', 'fwhm', 'magpsf', 'sigmapsf', 'ra',
            'dec', 'diffmaglim', 'rb', 'distnr', 'magnr', 'classtar',
            'ndethist', 'ncovhist', 'ecl_lat', 'ecl_long', 'gal_lat',
            'gal_long', 'non_detections', 'chinr', 'sharpnr']

#data_simclr = {'Train':{'images':[], 'features':[]}, 'Validation': data_red['Validation'], 'Test': data_red['Test']}
data_simclr = {'Train':{'images':[]}, 'Validation': data['Validation'], 'Test': data['Test']}

for path in os.listdir("/home/shared/pickles/"):
    
    if (".pickle" in path):

        print(path)
        df = pd.read_pickle("/home/shared/pickles/" + path)
    
        for index, row in df.iterrows():
          
            img = np.stack((row['cutoutScience'], row['cutoutTemplate'], row['cutoutDifference']),  axis=-1)
            stamp = repair_non_squared_stamp(img, row['xpos'], row['ypos'])
            stamp = preprocess_stamps(stamp, nan_val=0)

            data_simclr['Train']['images'].append(stamp)
            #data_simclr['Train']['features'].append([row[feature] for feature in features])

# -----------------------------------------------------------------------------

# Save dataset        
file = open('dataset/td_ztf_stamp_simclr_300.pkl', 'wb')
pickle.dump(data_simclr, file)
file.close()

# -----------------------------------------------------------------------------
