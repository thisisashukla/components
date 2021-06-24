import cv2
import numpy as np
import pandas as pd


def imageToDF(images, labels):
    """
    - DESCRIPTION
    creates a dataframe for mean and standard deviation of each channel of a list of images
    - ARGUMENTS
    images = list of image paths
    labels = list of label for images
    - OUTPUT
    df = dataframe containing rmean, rstd, gmean, gstd, bmean, bstd and label as columns
    rmean and rstd refers to mean and standard deviation of red channel.
    """
    for path, label in tqdm_notebook(zip(images, labels), total = len(images)):
        image = cv2.imread('{}.png'.format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        row = {}
        row['rmean'] = np.mean(image[:,:,0])
        row['rstd'] = np.std(image[:,:,0])
        row['gmean'] = np.mean(image[:,:,1])
        row['gstd'] = np.std(image[:,:,1])
        row['bmean'] = np.mean(image[:,:,2])
        row['bstd'] = np.std(image[:,:,2])
        row['label'] = label

        df.append(row, ignore_index=True)

    return df
