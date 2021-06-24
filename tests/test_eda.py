import pytest
import sys
sys.path.append('../')

from py_scripts.eda import Explore

@pytest.mark.explore
def test_num_describe(iris_data):

    exp = Explore(iris_data)
    descr = iris_data.describe()
    num_descr = exp.numDescribe()

    assert num_descr.shape[1] == descr.shape[1]
    
    print(num_descr)

@pytest.mark.explore
def test_cat_describe(iris_data):

    exp = Explore(iris_data)
    num_descr = exp.numDescribe()
    cat_descr = exp.catDescribe()
    num_cols = exp.num_cols
    cat_cols = exp.cat_cols
    
    print(num_descr)
    print('---')
    print(cat_descr)
    
    assert set(num_cols + cat_cols) == set(iris_data.columns.tolist())
    assert iris_data.describe().shape[1] == num_descr.shape[1]
    assert iris_data.shape[1] - iris_data.describe().shape[1] == cat_descr.shape[1]
    
    