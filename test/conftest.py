import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

@pytest.fixture
def iris_data():

    iris = load_iris()
    data = iris['data']
    target = iris['target']

    df = pd.DataFrame(data = data, columns = iris['feature_names'])
    df['target'] = np.apply_along_axis(lambda x: iris['target_names'][x], 0, target)
    
    return df
