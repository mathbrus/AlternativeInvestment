from collections import Counter
from datetime import datetime
import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import pickle
from scipy import isnan, stats
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import warnings


def transform_dataframe(data_frame, k=0, chunk_size=0, variable='RET', rem_init_index=0):
    """ This function basically extracts the data from the WRDS csv and
    puts it in a easy to use dataframe, where each company has its own column.
    k is the number of the current_chunk"""
    # get a list of the company PERMNOs and get the number of dates
    permno = data_frame.PERMNO.unique().tolist()
    all_dates = data_frame.date.unique().tolist()
    nb_comp = len(permno)  # Was used when we merged all the companies at once
    # We start by defining the values for k and chunk in case it is not specified
    if chunk_size == 0:
        chunk_size = nb_comp
        k = 0
    elif k > int(len(data_frame.PERMNO.unique().tolist()) / chunk_size):
        print("Wrong k")
        raise ValueError
    # create the dataframe for the transformed data containing only the dates for now
    extracted_data = pd.DataFrame(all_dates, columns=['date'])
    # the subset of the big dataframe, which we use to merge with the extracted_data dataframe
    next_comp_init = data_frame[['date', variable]]
    # We ignore the settingcopy warning since it is a false positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # If we are in the remainder we need to have rem_init_index != 0
        for i in range((k*chunk_size)+rem_init_index, ((k+1)*chunk_size)+rem_init_index):
            next_comp = next_comp_init.loc[data_frame.PERMNO == permno[i]]
            # we remove the C and B values and change them to NaN
            next_comp[variable] = next_comp[variable].replace(['C', 'B'], np.nan)
            extracted_data = extracted_data.merge(next_comp, how='left', on='date')
    # we finally create the DataFrame with all the returns
    table = pd.DataFrame(extracted_data.values, columns=['date'] + permno[(k*chunk_size)+rem_init_index:
                                                                          ((k+1)*chunk_size)+rem_init_index])
    return table


def transform_large_dataframe(data, chunk_size, variable='RET'):
    """Use this function instead of transform_dataframe if the imported data is large."""
    print("TRANSFORM LARGE DATAFRAME")
    # The compute the number of "full" chunks
    nb_comp = len(data.PERMNO.unique().tolist())
    nb_chunks = int(nb_comp/chunk_size)
    # Initialization of the big dataframe that we will return, with the date column
    big_table = pd.DataFrame(data.date.unique().tolist(), columns=['date'])
    tstart = datetime.now()
    # Processing each chunk
    for i in range(nb_chunks):
        transformed_subtable = transform_dataframe(data, k=i, chunk_size=chunk_size, variable=variable)
        transformed_subtable['date'] = transformed_subtable['date'].apply(int)
        big_table = big_table.merge(transformed_subtable, how='left', on='date')
        tinter = datetime.now() - tstart
        print("{}/{} time: {}".format(i+1, nb_chunks, tinter.total_seconds()))

    # We still miss the remainder, not included in the integer number of chunks
    print("Working on remainder")
    nb_comp_left = nb_comp - nb_chunks * chunk_size
    # We add the companies one by one starting from the first company not included
    remainder_subtable = transform_dataframe(data, k=0, chunk_size=nb_comp_left, variable=variable,
                                             rem_init_index=nb_chunks*chunk_size)
    remainder_subtable['date'] = remainder_subtable['date'].apply(int)
    big_table = big_table.merge(remainder_subtable, how='left', on='date')
    print("Remainder included")

    return big_table


def get_future_returns(df, permno, nb_months=7):
    """Data pre-processing for easy ML application. This function gives future returns that a stock will have made
    between today and (today + i). The input df has to be treated by transform_dataframe"""
    # The input dataframe has te be passed to transform_dataframe first.
    future_returns = pd.DataFrame(df.date.unique().tolist(), columns=['date'])
    # The 1d column is the return btw today and tomorrow
    future_returns['1m'] = df[permno].shift(-1).values
    incr = np.ones(len(df.index.unique().tolist()))
    # For the next days we compound the daily returns
    for i in range(2, nb_months+1):
        # Returns from WRDS CRSP are simple returns.
        future_returns['{}m'.format(i)] = (incr + df[permno].shift(-i).values.astype(float)) * \
                                          (incr + future_returns['{}m'.format(i-1)].values.astype(float)) - incr

    future_returns.set_index(['date'], inplace=True)
    return future_returns


def labelize_bsh(*args, threshold=0.03):
    """This is the simple function that puts a label on the future returns that we have.
    The initial implementation only gives buy sell or hold based on future returns"""
    future_returns_cols = [c for c in args]
    for future_returns_col in future_returns_cols:
        if future_returns_col > threshold:
            return 1
        if future_returns_col < -threshold:
            return -1
        return 0


def get_targets(df):
    """This function will create the targets for each date, based on the future returns
    for the next 7 days, and pass them to labelize_bsh. The input df has to be a transformed dataframe."""
    print("GET TARGETS")
    # We initalize the final output dataframe
    targets_df = pd.DataFrame(sorted(df.date.unique().tolist()), columns=['date'])
    targets_df.set_index(['date'], inplace=True)
    
    # We map the labelize_bsh function as a column of the newly created df, for each permno
    permno = df.columns.values.tolist()
    permno = permno[1:]
    tstart = datetime.now()
    for i in range(len(permno)):
        fut_ret = get_future_returns(df, permno[i], nb_months=7)
        targets_df[permno[i]] = list(map(labelize_bsh,
                                         fut_ret['1m'].astype(float),
                                         fut_ret['2m'].astype(float),
                                         fut_ret['3m'].astype(float),
                                         fut_ret['4m'].astype(float),
                                         fut_ret['5m'].astype(float),
                                         fut_ret['6m'].astype(float),
                                         fut_ret['7m'].astype(float)))
        if i % 320 == 0:
            print("{}%".format(round(i / len(permno) * 100)))
    tend = datetime.now() - tstart
    print("time: {}".format(tend.total_seconds()))
    return targets_df


def get_past_returns(df):
    """This function will calculate the cumulative return of an asset over the past 11 months, but not including
    the current month's return, to better separate momentum from reverse momentum.
    The only input is the transformer return df, the output has the same shape."""

    print("GET_PAST_RETURNS")
    # We create the function that computes the cumret for a specific column

    def get_past_returns_col(column):
        # Adding one to get factors
        col = (column.astype("float32")+1).values.tolist()
        # We have to use slices in order to point to a new object
        cumcol = col[:]
        # No built-in product function

        def prod(iterable):
            return functools.reduce(operator.mul, iterable, 1)

        # For the first eleven columns we have to do the cumret on less values
        # The last truncated cumret is at i = 10,
        if isnan(col[0]):
            col[0] = 1
        for i in range(1, 11):
            if isnan(col[i]):
                col[i] = 1
            cumcol[i] = prod(col[0:i])

        # After that we do on the 11 values before
        for i in range(11, len(col)):
            if isnan(col[i]):
                col[i] = 1
            cumcol[i] = prod(col[i-11:i])

        cumcol[0] = 1
        cumcol[:] = [x - 1 for x in cumcol]
        return cumcol

    # We now apply the to every column of the returns, excluding the dates
    df = df[:]
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(get_past_returns_col, axis=0)

    return df


def ml_dataframe(y_df, ret_df, permno_list, x1_df, x2_df):
    """This function will prepare the dataframe on which we can do actual ML. The output format resembles
    the one from the initial WRDS input. For every target (first column), we can learn to predict it with 
    the data in the following columns from the same row. ret_df is used for logical indexing, so as to
    filter out all the NaNs."""

    # We initialize the dataframe that we want to return
    ml_df = pd.DataFrame({'date': [19509658], 'PERMNO': [00000], 'target': [0], 'price_1m': [0], 'price_122': [0]})
    
    for i in range(len(permno_list)):
        # We start by finding the portion of the targets which corresponds to actual returns.
        y_df_subtable = y_df[permno_list[i]]
        y_df_subtable = y_df_subtable[~isnan(ret_df[permno_list[i]])]

        # x1 subtable
        x1_df_subtable = x1_df[permno_list[i]]
        x1_df_subtable = x1_df_subtable[~isnan(ret_df[permno_list[i]])]

        # x2 subtable
        x2_df_subtable = x2_df[permno_list[i]]
        x2_df_subtable = x2_df_subtable[~isnan(ret_df[permno_list[i]])]

        # Resetting index for merger and concatenation
        y_df_subtable = y_df_subtable.reset_index()
        x1_df_subtable = x1_df_subtable.reset_index()
        x2_df_subtable = x2_df_subtable.reset_index()
        # Renaming columns
        y_df_subtable = y_df_subtable.rename(index=str, columns={"date": "date", permno_list[i]: "target", })
        x1_df_subtable = x1_df_subtable.rename(index=str, columns={"date": "date", permno_list[i]: "price_1m", })
        x2_df_subtable = x2_df_subtable.rename(index=str, columns={"date": "date", permno_list[i]: "price_122", })
        # The column of permnos
        y_df_subtable = y_df_subtable.assign(PERMNO=[permno_list[i] for t in range(len(y_df_subtable))])
        # Merging
        y_df_subtable = y_df_subtable.merge(x1_df_subtable, how='left', on='date')
        y_df_subtable = y_df_subtable.merge(x2_df_subtable, how='left', on='date')
        # We append, but without the last value since there is no future price afterwards.
        # And without the first value since there is no TwelveToTwo past return
        ml_df = ml_df.append(y_df_subtable[1:-1], ignore_index=True, sort=False)

    # Deleting the first, useless value
    ml_df = ml_df.drop(ml_df.index[0])
        
    return ml_df


def large_ml_dataframe(y_df, ret_df, x1_df, x2_df, chunk_size=320):
    """Similarly as with the transform_large_dataframe, we need to divide the process into chunks.
    As inputs, y_df is indexed on dates, the other dataframes are not."""
    print("LARGE ML DATAFRAME")

    permno = ret_df.columns.values.tolist()
    permno = permno[1:]

    # We set indices on date for concatenation and logical indexing purposes
    ret_df = ret_df.set_index(['date'])
    x1_df = x1_df.set_index(['date'])
    x2_df = x2_df.set_index(['date'])
    # We transform them in a numeric dataframe instead of an object
    ret_df = pd.DataFrame(ret_df, dtype="float32")
    x1_df = pd.DataFrame(x1_df, dtype="float32")
    x2_df = pd.DataFrame(x2_df, dtype="float32")

    nb_chunks = round(len(permno) / chunk_size)

    # We initialize the dataframe that we want to return
    large_ml_df = pd.DataFrame({'date': [19509658], 'PERMNO': [00000], 'target': [0], 'price_1m': [0],
                                'price_122': [0]})
    
    tstart = datetime.now()
    for k in range(nb_chunks):
        chunk_df = ml_dataframe(y_df=y_df, ret_df=ret_df, permno_list=permno[k*chunk_size:(k+1)*chunk_size],
                                x1_df=x1_df, x2_df=x2_df)
        large_ml_df = large_ml_df.append(chunk_df, ignore_index=True, sort=False)
        tinter = datetime.now() - tstart
        print("{}/{} time: {}".format(k+1, nb_chunks, tinter.total_seconds()))
     
    # Remainder    
    rem_df = ml_dataframe(y_df=y_df, ret_df=ret_df, permno_list=permno[nb_chunks*chunk_size:],
                          x1_df=x1_df, x2_df=x2_df)
    large_ml_df = large_ml_df.append(rem_df, ignore_index=True, sort=False)
    # We remove the first, useless value
    large_ml_df = large_ml_df.drop(0)
    large_ml_df = large_ml_df.reset_index(drop=True)
    return large_ml_df


def standardize(ml_dataframe):
    """This function normalizes the X columns of the ml_dataframe. Note that since we use iloc, which is not a copy,
    it directly modifies the original ml_dataframe."""
    mean = ml_dataframe.iloc[:, 3:].mean()
    std = ml_dataframe.iloc[:, 3:].std()
    ml_dataframe.iloc[:, 3:] = (ml_dataframe.iloc[:, 3:]-mean)/std


def fit_knn(ml_dataframe):

    print("FITTING KNN")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X, y)

    # outfile = open('pickle/full/naive/KNN', 'wb')
    outfile = open('pickle/full/CV/KNN', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))


def fit_svc(ml_dataframe):

    print("FITTING SVC")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = svm.LinearSVC()
    clf.fit(X, y)

    # outfile = open('pickle/full/naive/SVC', 'wb')
    outfile = open('pickle/full/CV/SVC', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))


def fit_rfc(ml_dataframe, max_depth=1000):

    print("FITTING RFC")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
    clf.fit(X, y)

    # outfile = open('pickle/full/naive/RFC', 'wb')
    outfile = open('pickle/full/CV/RFC', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))


def fit_lr(ml_dataframe):

    print("FITTING LR")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = LogisticRegression(solver="saga", multi_class="ovr", n_jobs=-1)
    clf.fit(X, y)

    # outfile = open('pickle/full/naive/LR', 'wb')
    outfile = open('pickle/full/CV/LR', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))


def predict_knn(ml_dataframe):

    print("PREDICTING KNN")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']

    # infile = open('pickle/full/naive/KNN', 'rb')
    infile = open('pickle/full/CV/KNN', 'rb')
    clf = pickle.load(infile)
    infile.close()

    predictions = clf.predict(X)
    confidence = clf.score(X, y)
    print("Spread : ", Counter(predictions))
    print("Confidence : ", confidence)

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))

    return predictions


def predict_svc(ml_dataframe):

    print("PREDICTING SVC")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']

    # infile = open('pickle/full/naive/SVC', 'rb')
    infile = open('pickle/full/CV/SVC', 'rb')
    clf = pickle.load(infile)
    infile.close()

    predictions = clf.predict(X)
    confidence = clf.score(X, y)
    print("Spread : ", Counter(predictions))
    print("Confidence : ", confidence)

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))

    return predictions


def predict_rfc(ml_dataframe):

    print("PREDICTING RFC")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']

    # infile = open('pickle/full/naive/RFC', 'rb')
    infile = open('pickle/full/CV/RFC', 'rb')
    clf = pickle.load(infile)
    infile.close()

    predictions = clf.predict(X)
    confidence = clf.score(X, y)
    print("Spread : ", Counter(predictions))
    print("Confidence : ", confidence)

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))

    return predictions


def predict_lr(ml_dataframe):

    print("PREDICTING LR")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']

    # infile = open('pickle/full/naive/LR', 'rb')
    infile = open('pickle/full/CV/LR', 'rb')
    clf = pickle.load(infile)
    infile.close()

    predictions = clf.predict(X)
    confidence = clf.score(X, y)
    print("Spread : ", Counter(predictions))
    print("Confidence : ", confidence)

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))

    return predictions


def aggregate_prediction(knn, svc, rfc, lr):
    """This function makes the democratic vote of the different classifiers"""
    print("COMPUTING AGGREGATE PREDICTIONS")
    votes_sum = knn+svc+rfc+lr
    output = votes_sum[:]
    for i in range(len(votes_sum)):
        if votes_sum[i] <= -2:
            output[i] = -1
        elif -1 <= votes_sum[i] <= 1:
            output[i] = 0
        elif votes_sum[i] >= 2:
            output[i] = 1

    return output


def aggregate_accuracy(agg_pred, ml_dataframe):
    """This function gives the accuracy of our aggregate predictions."""
    print("COMPUTING AGGREGATE ACCURACY")
    counter = 0
    for i in range(len(agg_pred)):
        if agg_pred[i] == ml_dataframe.target[i]:
            counter += 1

    print("Target distribution : ", Counter(ml_dataframe.target))
    print("Aggragate prediction : ", Counter(agg_pred))
    print("Accuracy : {}".format(counter/len(agg_pred)))


def portfolio_performance(transformed_dataframe_returns, transformed_dataframe_mcap, prediction_transformed_dataframe):
    """This is the big function that will evaluate the performance of our investment strategy.
    The inputs are simply the returns and market caps of the whole dataset, and the predicted choice of stocks.
    The output is the time-series of returns of the formed long-short portfolio."""
    tstart = datetime.now()
    # We start by replacing all the NaNs by 0, since not investing or not existing is essentially the same here
    prediction_transformed_dataframe = \
        prediction_transformed_dataframe.where(~isnan(prediction_transformed_dataframe), other=0)
    # We now change all the values to int, for faster comparison
    prediction_transformed_dataframe = prediction_transformed_dataframe.astype("int64")


    # We initialize the weights matrices, and set to 1 all stocks that we invest in.
    # There are two portfolios here, one long and one short. We will subtract the returns later on.
    weights_long = prediction_transformed_dataframe.copy()
    weights_long.where(~(weights_long < 1), other=0, inplace=True)

    weights_short = prediction_transformed_dataframe.copy()
    weights_short.where(~(weights_short > -1), other=0, inplace=True)
    weights_short.where(~(weights_short == -1), other=1, inplace=True)
    weights_short.date = weights_long.date # We reset the dates since they have disappeared in the previous operation.

    # There is one subtlety : investment decision is based on past info, so if the stock does
    # not exist anymore in the next month, then we will not be able to invest.

    # We shift the index of the returns by 2. One because we have no prediction for the first month
    # and one so as to shift the returns by 1 month and compare to investment decisions
    shifted_transformed_dataframe_returns = transformed_dataframe_returns.iloc[2:].copy()
    shifted_transformed_dataframe_returns = shifted_transformed_dataframe_returns.reset_index(drop=True)
    # We make sure that the returns df is of native dtype, to use the isna function later
    shifted_transformed_dataframe_returns = shifted_transformed_dataframe_returns.astype("float64")
    # We set the columns to be the same as the weights matrices, for faster logical indexing
    shifted_transformed_dataframe_returns.set_axis(weights_long.columns, axis='columns', inplace=True)

    # Note that shifted_... and weights_... have the same shape
    # We set to 0 investment decisions where there is no return in the next month.
    weights_long.where(~shifted_transformed_dataframe_returns.isna(), other=0, inplace=True)
    weights_short.where(~shifted_transformed_dataframe_returns.isna(), other=0, inplace=True)

    # We now need to find the respective weights of all stocks, using a value-weighted approach.
    # We calculate the relative weight of each chosen stock for each date

    # Taking absolute value because certain stocks have neg price & resetting index for division
    transformed_dataframe_mcap = transformed_dataframe_mcap.iloc[1:-1, 1:].copy().reset_index(drop=True).abs()
    # Setting to 0 all mcaps of stocks that are not selected
    transformed_dataframe_mcap_long = transformed_dataframe_mcap.copy()
    transformed_dataframe_mcap_short = transformed_dataframe_mcap.copy()
    transformed_dataframe_mcap_long.set_axis(weights_long.columns[1:], axis='columns', inplace=True)
    transformed_dataframe_mcap_short.set_axis(weights_long.columns[1:], axis='columns', inplace=True)
    transformed_dataframe_mcap_long.where(~(weights_long.iloc[:, 1:] == 0), other=0, inplace=True)
    transformed_dataframe_mcap_short.where(~(weights_short.iloc[:, 1:] == 0), other=0, inplace=True)

    # We include a measure for the transaction costs
    # We first look at the number of transactions
    # It works the following way : we create a copy of each weight matrix and shift it by 1 period
    # The shifted matrix is multiplied by 2 and then add it to the original weight matrix
    # If the sum is -3, 0 or 3 , then weights have not changed. Otherwise we have a tx cost.

    weights_long_shifted = weights_long.iloc[1:, 1:].copy().reset_index(drop=True)
    weights_long_combined = weights_long_shifted*2 + weights_long.iloc[:-1, 1:]

    weights_short_shifted = weights_short.iloc[1:, 1:].copy().reset_index(drop=True)
    weights_short_combined = weights_short_shifted*2 + weights_short.iloc[:-1, 1:]

    tx_costs_weights_long = weights_long_combined.copy()  # just for the shape
    tx_costs_weights_long[:] = 1
    tx_costs_weights_short = weights_short_combined.copy()  # just for the shape
    tx_costs_weights_short[:] = 1

    tx_costs_weights_long.where(~(weights_long_combined == 3), other=0, inplace=True)
    tx_costs_weights_long.where(~(weights_long_combined == 0), other=0, inplace=True)
    tx_costs_weights_long.where(~(weights_long_combined == -3), other=0, inplace=True)

    tx_costs_weights_short.where(~(weights_short_combined == 3), other=0, inplace=True)
    tx_costs_weights_short.where(~(weights_short_combined == 0), other=0, inplace=True)
    tx_costs_weights_short.where(~(weights_short_combined == -3), other=0, inplace=True)

    nb_tx_long = tx_costs_weights_long.sum(axis=1)
    nb_tx_short = tx_costs_weights_short.sum(axis=1)
    # We calculate the market-cap-weighted weight of each stock, aka value-weighted approach
    weights_long.iloc[:, 1:] = transformed_dataframe_mcap_long.divide(transformed_dataframe_mcap_long.sum(axis=1),
                                                                      axis='index')
    weights_short.iloc[:, 1:] = transformed_dataframe_mcap_short.divide(transformed_dataframe_mcap_short.sum(axis=1),
                                                                        axis='index')

    # We then calculate the difference in weights btw 2 periods, which also impacts the tx costs
    # This has to be done after the value-weighting
    weights_long_shifted = weights_long.iloc[1:, 1:].copy().reset_index(drop=True)
    weights_short_shifted = weights_short.iloc[1:, 1:].copy().reset_index(drop=True)

    weights_long_turnover = (weights_long_shifted - weights_long.iloc[:-1, 1:]).abs()
    weights_short_turnover = (weights_short_shifted - weights_short.iloc[:-1, 1:]).abs()

    # Gives us the period turnover ratio
    turnover_long = weights_long_turnover.sum(axis=1)
    turnover_short = weights_short_turnover.sum(axis=1)
    # Remark : the combined change of absolute weights can be maximum 2 in case of "complete turnover"

    # we calculate the tx costs in terms of percentage
    tx_costs_long = (1+turnover_long)*nb_tx_long
    tx_costs_short = (1+turnover_short)*nb_tx_short

    # We now want to calculate the performance of our two portfolios
    returns_df_long = shifted_transformed_dataframe_returns.iloc[:, 1:].multiply(weights_long.iloc[:, 1:], fill_value=0)
    returns_df_short = shifted_transformed_dataframe_returns.iloc[:, 1:].multiply(weights_short.iloc[:, 1:],
                                                                                  fill_value=0)
    # We compute the return for each date as the sum of the elements of that row (weighted sum)
    performance_long = returns_df_long.sum(axis=1)
    performance_short = returns_df_short.sum(axis=1)

    performance_series = (performance_long-performance_short)

    return performance_series, tx_costs_long, tx_costs_short


def performance_analysis(portfolio_series):
    """This function will derive several things from the portfolio returns,
    The input is the portfolio returns and the outputs can be seen in the print statement."""
    # Importing Fama-French 3 factors
    fama_raw = pd.read_csv("fama_french3.csv")
    # Extracting the dates we want and dividing by 100 since Ken French data is in "full" percentage
    # We do not include the first and last value of our period since we have no portfolio at that time
    # fama_french_factors = fama_raw.loc[(fama_raw["date"] > 196307) & (fama_raw["date"] < 201806),
    #                                    ["date", "Mkt-RF", "SMB", "HML", "RF"]].reset_index(drop=True)/100
    fama_french_factors = fama_raw.loc[(fama_raw["date"] > 200001) & (fama_raw["date"] < 201806),
                                       ["date", "Mkt-RF", "SMB", "HML", "RF"]].reset_index(drop=True) / 100

    # Total return
    # No built-in product function
    def prod(iterable):
        return functools.reduce(operator.mul, iterable, 1)
    tot_ret = prod(portfolio_series+1)-1

    # Average return
    avg_ret = portfolio_series.mean()

    # Monthly standard deviation
    std = portfolio_series.std()

    # Sharpe Ratio
    monthly_sharpe_ratio = (portfolio_series-fama_french_factors["RF"]).mean()\
        / (portfolio_series-fama_french_factors["RF"]).std()
    yearly_sharpe_ratio = monthly_sharpe_ratio*math.sqrt(12)  # Shortcup implies normal returns -> not very realistic

    # Alpha/Betas, including significance tests - from stackexchange
    X = fama_french_factors[['Mkt-RF', 'SMB', 'HML']]
    y = portfolio_series
    lm = LinearRegression()
    lm.fit(X, y)
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)

    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    mse = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = mse * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    # Maximum Drawdown - from Stackexchange

    def max_drawdown(X):
        mdd = 0
        peak = X[0]
        for x in X:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return mdd
    max_dd = max_drawdown((1+portfolio_series).cumprod())

    # Graphing the cumulative product of returns
    plt.plot(fama_french_factors["date"].astype(str), (1+portfolio_series).cumprod())
    plt.legend(['Portfolio'])
    plt.xticks(rotation=90)
    # Show one date every two year only
    plt.xticks(np.arange(0, len(portfolio_series), step=round(len(portfolio_series)/12)))
    # plt.xticks(np.arange(0, 120, step=10))
    plt.yscale('log')
    plt.savefig('perf_CV.png')
    plt.show()


    summary = pd.DataFrame()
    summary["##"] = ["Total Return", "Average Monthly Return", "Monthly Standard deviation",
                     "Monthly Sharpe Ratio", "Annualized Sharpe Ratio", "Alpha", "Alpha P-value", "Maximum Drawdown"]
    summary["Values"] = [tot_ret, avg_ret, std, monthly_sharpe_ratio, yearly_sharpe_ratio,
                         params[0], p_values[0], max_dd]
    summary.set_index("##", inplace=True)

    print(summary)

    return summary


# # ------------------------------------------------------------------------------------------------------------------
# NAIVE IN-SAMPLE

# print("-----------------------START OF LOG-----------------------------")
# print("")
# t_total_start = datetime.now()
#
# data = pd.read_csv("raw_data_full.csv")
# tdf_ret = transform_large_dataframe(data, chunk_size=200)
# tdf_shrout = transform_large_dataframe(data, variable='SHROUT', chunk_size=200)
# tdf_prc = transform_large_dataframe(data, variable='PRC', chunk_size=200)
#
# # Because of the shuffled dates
#
# tdf_ret = tdf_ret.set_index(['date'])
# tdf_ret = tdf_ret.sort_index()
# tdf_ret = tdf_ret.reset_index()
# tdf_shrout = tdf_shrout.set_index(['date'])
# tdf_prc = tdf_prc.set_index(['date'])
# tdf_prc = tdf_prc.sort_index()
# tdf_shrout = tdf_shrout.sort_index()
# tdf_prc = tdf_prc.reset_index()
# tdf_shrout = tdf_shrout.reset_index()
#
# # tdf_shrout = pd.read_csv("CSV/tdf_shrout.csv", index_col=0)
# # tdf_prc = pd.read_csv("CSV/tdf_prc.csv", index_col=0)
#
# #
# # Obtaining the market caps
#
# tdf_mcap = tdf_shrout.copy()
# tdf_mcap.iloc[:, 1:] = tdf_shrout.iloc[:, 1:]*tdf_prc.iloc[:, 1:]
#
# targets_df = get_targets(tdf_ret)
# targets_df.to_csv("CSV/targets_df.csv")
# cum_df = get_past_returns(tdf_ret)
# cum_df.to_csv("CSV/cum_df.csv")
# cl = large_ml_dataframe(targets_df, tdf_ret, tdf_ret, cum_df, chunk_size=200)

# # cl.to_csv("CSV/cl.csv")
# # cl = pd.read_csv("CSV/cl.csv", index_col=0)

# standardize(cl)
#
# fit_knn(cl)
# knn = predict_knn(cl)
# fit_svc(cl)
# svc = predict_svc(cl)
# fit_rfc(cl, 20)
# rfc = predict_rfc(cl)
# fit_lr(cl)
# lr = predict_lr(cl)
#
# aggpred = aggregate_prediction(knn, svc, rfc, lr)
# aggregate_accuracy(aggpred, cl)
#
# pred_column = pd.DataFrame(aggpred, columns=['prediction'])
# cl = cl.join(pred_column, sort=False)
# new_tdf = transform_large_dataframe(cl, chunk_size=200, variable='prediction')
#
# new_tdf = new_tdf.set_index(['date'])
# new_tdf = new_tdf.sort_index()
# new_tdf = new_tdf.reset_index()
#
# outfile = open('pickle/temp/new_tdf', 'wb')
# pickle.dump(new_tdf, outfile)
# outfile.close()
#
# # Because of the companies that we have to drop
# diff = list(set(tdf_ret.columns[1:].astype("str")) - set(new_tdf.columns[1:].astype("int32").astype("str")))
#
# #
# performance, tx_cost_long, tx_cost_short = portfolio_performance(tdf_ret.drop(diff, axis=1),
#                                                                      tdf_mcap.drop(diff, axis=1), new_tdf)
#
# outfile = open('pickle/temp/ind_3_naive_perf', 'wb')
# pickle.dump(performance, outfile)
# outfile.close()
# outfile = open('pickle/temp/ind_3_naive_txcl', 'wb')
# pickle.dump(tx_cost_long, outfile)
# outfile.close()
# outfile = open('pickle/temp/ind_3_naive_txcs', 'wb')
# pickle.dump(tx_cost_short, outfile)
# outfile.close()
#
# performance_series = performance.sub(tx_cost_long*0.000005, fill_value=0)
# performance_series = performance_series.sub(tx_cost_short*0.000005, fill_value=0)
# perf = performance_analysis(performance_series)
#
# print("")
# print("-----------------------END OF LOG-----------------------------")
# print("")
# t_total_end = datetime.now() - t_total_start
# print("Total Elapsed Time : {} seconds.".format(t_total_end.total_seconds()))
#
#
#
# # ------------------------------------------------------------------------------------------------------------------
# CROSS-VALIDATION

# print("-----------------------START OF LOG-----------------------------")
# print("")
# t_total_start = datetime.now()
#
# data = pd.read_csv("raw_data_full.csv")
# tdf_ret = transform_large_dataframe(data, chunk_size=200)
# tdf_shrout = transform_large_dataframe(data, variable='SHROUT', chunk_size=200)
# tdf_prc = transform_large_dataframe(data, variable='PRC', chunk_size=200)
#
# # Because of the shuffled dates
#
# tdf_ret = tdf_ret.set_index(['date'])
# tdf_ret = tdf_ret.sort_index()
# tdf_ret = tdf_ret.reset_index()
# tdf_shrout = tdf_shrout.set_index(['date'])
# tdf_prc = tdf_prc.set_index(['date'])
# tdf_prc = tdf_prc.sort_index()
# tdf_shrout = tdf_shrout.sort_index()
# tdf_prc = tdf_prc.reset_index()
# tdf_shrout = tdf_shrout.reset_index()
#
# # tdf_shrout = pd.read_csv("CSV/tdf_shrout.csv", index_col=0)
# # tdf_prc = pd.read_csv("CSV/tdf_prc.csv", index_col=0)
#
# #
# # Obtaining the market caps
#
# tdf_mcap = tdf_shrout.copy()
# tdf_mcap.iloc[:, 1:] = tdf_shrout.iloc[:, 1:]*tdf_prc.iloc[:, 1:]
#
# targets_df = get_targets(tdf_ret)
# targets_df.to_csv("CSV/targets_df.csv")
# cum_df = get_past_returns(tdf_ret)
# cum_df.to_csv("CSV/cum_df.csv")
#
# cl_train = large_ml_dataframe(targets_df.iloc[:438, :], tdf_ret.iloc[:438, :], tdf_ret.iloc[:438, :], cum_df.iloc[:438, :], chunk_size=200)
# cl_test = large_ml_dataframe(targets_df.iloc[438:, :], tdf_ret.iloc[438:, :], tdf_ret.iloc[438:, :], cum_df.iloc[438:, :], chunk_size=200)
#
#
# standardize(cl_train)
# standardize(cl_test)
#
# fit_knn(cl_train)
# knn = predict_knn(cl_test)
# fit_svc(cl_train)
# svc = predict_svc(cl_test)
# fit_rfc(cl_train, 20)
# rfc = predict_rfc(cl_test)
# fit_lr(cl_train)
# lr = predict_lr(cl_test)
#
# aggpred = aggregate_prediction(knn, svc, rfc, lr)
# aggregate_accuracy(aggpred, cl_test)
#
# pred_column = pd.DataFrame(aggpred, columns=['prediction'])
# cl_test = cl_test.join(pred_column, sort=False)
# new_tdf = transform_large_dataframe(cl_test, chunk_size=200, variable='prediction')
#
# new_tdf = new_tdf.set_index(['date'])
# new_tdf = new_tdf.sort_index()
# new_tdf = new_tdf.reset_index()
#
# outfile = open('pickle/temp/new_tdf', 'wb')
# pickle.dump(new_tdf, outfile)
# outfile.close()
#
# # Because of the companies that we have to drop
# diff = list(set(tdf_ret.columns[1:]) - set(new_tdf.columns[1:].astype("int32")))
# #
# performance, tx_cost_long, tx_cost_short = portfolio_performance(tdf_ret.iloc[438:, :].drop(diff, axis=1),
#                                                                      tdf_mcap.iloc[438:, :].drop(diff, axis=1), new_tdf)
#
#
# outfile = open('pickle/temp/ind_3_cv_perf', 'wb')
# pickle.dump(performance, outfile)
# outfile.close()
# outfile = open('pickle/temp/ind_3_cv_txcl', 'wb')
# pickle.dump(tx_cost_long, outfile)
# outfile.close()
# outfile = open('pickle/temp/ind_3_cv_txcs', 'wb')
# pickle.dump(tx_cost_short, outfile)
# outfile.close()
#
# performance_series = performance.sub(tx_cost_long*0.000005, fill_value=0)
# performance_series = performance_series.sub(tx_cost_short*0.000005, fill_value=0)
# perf = performance_analysis(performance)
#
# print("")
# print("-----------------------END OF LOG-----------------------------")
# print("")
# t_total_end = datetime.now() - t_total_start
# print("Total Elapsed Time : {} seconds.".format(t_total_end.total_seconds()))




# ------------------------------------------------------------------------------------------------------------------
# The medium dataset

# print("-----------------------START OF LOG medium-----------------------------")
# print("")
# t_total_start = datetime.now()
#
# # data = pd.read_csv("little_test_data.csv")
# data = pd.read_csv("raw_data_full.csv")
# data = data.iloc[:12000, :]
# tdf_ret = transform_large_dataframe(data, chunk_size=51)
# tdf_shrout = transform_large_dataframe(data, variable='SHROUT', chunk_size=51)
# tdf_prc = transform_large_dataframe(data, variable='PRC', chunk_size=51)
#
# # Because of the shuffled dates
#
# tdf_ret = tdf_ret.set_index(['date'])
# tdf_ret = tdf_ret.sort_index()
# tdf_ret = tdf_ret.reset_index()
# tdf_shrout = tdf_shrout.set_index(['date'])
# tdf_prc = tdf_prc.set_index(['date'])
# tdf_prc = tdf_prc.sort_index()
# tdf_shrout = tdf_shrout.sort_index()
# tdf_prc = tdf_prc.reset_index()
# tdf_shrout = tdf_shrout.reset_index()
#
# # tdf_shrout = pd.read_csv("CSV/tdf_shrout.csv", index_col=0)
# # tdf_prc = pd.read_csv("CSV/tdf_prc.csv", index_col=0)
#
# #
# # Obtaining the market caps
#
# tdf_mcap = tdf_shrout.copy()
# tdf_mcap.iloc[:, 1:] = tdf_shrout.iloc[:, 1:]*tdf_prc.iloc[:, 1:]
#
# targets_df = get_targets(tdf_ret)
# targets_df.to_csv("CSV/targets_df.csv")
# cum_df = get_past_returns(tdf_ret)
# cum_df.to_csv("CSV/cum_df.csv")
#
# cl_train = large_ml_dataframe(targets_df.iloc[:438, :], tdf_ret.iloc[:438, :], tdf_ret.iloc[:438, :], cum_df.iloc[:438, :], chunk_size=51)
# cl_test = large_ml_dataframe(targets_df.iloc[438:, :], tdf_ret.iloc[438:, :], tdf_ret.iloc[438:, :], cum_df.iloc[438:, :], chunk_size=51)
#
# # cl.to_csv("CSV/cl.csv")
# # cl = pd.read_csv("CSV/cl.csv", index_col=0)
#
# standardize(cl_train)
# standardize(cl_test)
#
# fit_knn(cl_train)
# knn = predict_knn(cl_test)
# fit_svc(cl_train)
# svc = predict_svc(cl_test)
# fit_rfc(cl_train, 10)
# rfc = predict_rfc(cl_test)
# fit_lr(cl_train)
# lr = predict_lr(cl_test)
#
# aggpred = aggregate_prediction(knn, svc, rfc, lr)
# aggregate_accuracy(aggpred, cl_test)
#
# pred_column = pd.DataFrame(aggpred, columns=['prediction'])
# cl_test = cl_test.join(pred_column, sort=False)
# new_tdf = transform_large_dataframe(cl_test, chunk_size=51, variable='prediction')
#
# new_tdf = new_tdf.set_index(['date'])
# new_tdf = new_tdf.sort_index()
# new_tdf = new_tdf.reset_index()
#
# outfile = open('pickle/temp/new_tdf', 'wb')
# pickle.dump(new_tdf, outfile)
# outfile.close()
#
# # Because of the companies that we have to drop
# diff = list(set(tdf_ret.columns[1:]) - set(new_tdf.columns[1:].astype("int32")))
# #
# performance, tx_cost_long, tx_cost_short = portfolio_performance(tdf_ret.iloc[438:, :].drop(diff, axis=1),
#                                                                      tdf_mcap.iloc[438:, :].drop(diff, axis=1), new_tdf)
#
#
# perf = performance_analysis(performance)
#
# print("")
# print("-----------------------END OF LOG-----------------------------")
# print("")
# t_total_end = datetime.now() - t_total_start
# print("Total Elapsed Time : {} seconds.".format(t_total_end.total_seconds()))