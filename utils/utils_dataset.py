import csv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

label_names = ['AGN', 'SN', 'VS', 'Asteroid', 'Bogus']

feature_names = ['sgscore_1', 'sgscore_2', 'sgscore_3', 'distpsnr_1',
                 'distpsnr_2', 'distpsnr_3', 'isdiffpos', 'fwhm', 'magpsf',
                 'sigmapsf', 'RA', 'DEC', 'diffmaglim', 'classtar', 'ndethist',
                 'ncovhist', 'chinr', 'sharpnr', 'Ecliptic coordinates RA',
                 'Ecliptic coordinates DEC', 'Galactic coordinates RA',
                 'Galactic coordinates DEC', 'approx non-detections']

# -----------------------------------------------------------------------------

# Recursively prints the type of the objects inside the input.
def dataset_structure(data, space=''):

    # Iteration along dictionary
    for keys in data:

        # If the object is other dictionary
        if(type(data[keys]) is dict):
            print('{0}-{1}:'.format(space,keys))
            dataset_structure(data[keys], space + '\t')

        else:

            # Type of the object
            type_obj = type(data[keys]).__name__

            # Extracts an example of the elements into the object
            example = data[keys][0]
            type_example = type(example).__name__

            # If the element is an ndarray, its shape is printed
            if(type_example=='ndarray'):
                shape_example = str(np.shape(example))
            else:
                shape_example = 'Non-ndarray'

            # Prints the information (length and type) of the structure inside
            # the dictionary
            print("{0}-{1:<10}N={2:<10}Type:{3}".format(space, keys, len(data[keys]), type_obj), end="")
            print("\t-->\tType:{1:<10}Shape:{2:<10}".format(space, type_example, shape_example))

# -----------------------------------------------------------------------------

## Returns a list with the array into the file
#
#def load_array(file):
#
#    csv_reader = csv.reader(open(file, "r"))
#    list_array = []
#
#    # Only loads the first element
#    for row in csv_reader:
#        list_array.append(row[0])
#
#    return list_array

# -----------------------------------------------------------------------------

# Recursively prints the type of the objects inside the input.
#def reduction_images(data, size):

#    for img in data['Train']['images']:

#    for img in data['Validation']['images']:

#    for img in data['Test']['images']: