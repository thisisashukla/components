import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook


def RolledUpCoocurrence(df, freq):
    """
    - DESCRIPTION
    generates co-occurrence matrix for every column pair in a given dataframe
    after rolling up the dataframe based on a given frequency of datetime
    - ARGUMENTS
    df = pandas dataframe with a datetime column timestamp
    freq = frequency of rollup. Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    - OUTPUT
    roll = rolled up data at the level of the given frequency
    countMat = co-occurrence matrix
    percentMat = percentage co-occurrence matrix
    """
    df['round'] = df.timestamp.dt.round(freq)
    roll = df.groupby(['round']).sum()
    countMat = pd.DataFrame(index =  np.arange(len(roll.columns)), columns = np.arange(len(roll.columns)))
    div = roll.shape[0]
    for i in tqdm(range(len(roll.columns)), total = len(roll.columns)):
        for j in range(i+1, len(roll.columns)):
            temp = roll.iloc[:,[i,j]].min(axis=1)
            count = sum(np.where(temp > 0, temp, 0))
            countMat.iloc[i,j] = count
            countMat.iloc[j,i] = count
    countMat.index = minuteroll.columns
    countMat.columns = minuteroll.columns
    percentMat = (countMat*100)/div

    return roll, countMat, percentMat


def corrTwoDF(df1, df2, var1, var2, agg1, agg2, roll_var, lag = 0):
    """
    - DESCRIPTION
    generates correlation matrix between two unique values variables from two dataframes
    aggregated with respect to respective variables and another common variable
    - ARGUMENTS
    df1 = first data frame with rollup variable, variable1 and aggegrate1 as columns
    df2 = second data frame with rollup variable, variable2 and aggegrate2 as columns
    var1 = variable one for aggregating aggregate1
    var2 = variable one for aggregating aggregate2
    agg1 = column which will be aggregated based on roll up variable and variable1 in df1
    agg2 = column which will be aggregated based on roll up variable and variable2 in df2
    roll_var = common pandas datetime variable between df1 and df2
    lag = days by which df1 should lag by df2
    - OUTPUT
    data1 = dataframe containing aggregated values of agg1 at each timestamp for
    unique values of var1 as columns
    data1 = dataframe containing aggregated values of agg2 at each timestamp for
    unique values of var2 as columns
    """
    grpby1 = df1[[roll_var, var1, agg1]].groupby([roll_var, var1]).sum().reset_index()
    data1 = pd.DataFrame(index = grpby1[roll_var].unique(), columns = df1[var1].unique())
    data1 = data1.fillna(0)

    for i, v in tqdm_notebook(data1.iterrows(), total = data1.shape[0]):
        subset = grpby1[grpby1[roll_var] == i]
        for j, w in subset.iterrows():
            data1.loc[i, w[var1]] = w[agg1]

    data2 = pd.DataFrame(index = data1.index, columns = df2[var2].unique())
    data2 = data2.fillna(0)

    for i,v in tqdm_notebook(data2.iterrows(), total = data2.shape[0]):
        subset = df2[(df2[roll_var] == (i + pd.Timedelta(days = lag)))]
        if(subset.shape[0] > 0):
            for j, w in subset.iterrows():
                data2.loc[i, w[var2]]+= subset[agg2].values[0]


    corr = pd.DataFrame(index = data2.columns, columns = data1.columns)
    for i in tqdm_notebook(corr.index):
        for j in corr.columns:
            corr.loc[i,j] = data2[i].corr(data1[j])
    corr = corr.fillna(0)

    return data1, data2, corr
