from collections import Counter
from datetime import datetime
import functools
import numpy as np
import operator
import pandas as pd
import pickle
from scipy import isnan
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings


def transform_dataframe(data_frame, k=0, chunk_size=0, variable='RET', rem_init_index=0):
    """ This function basically extracts the data from the WRDS csv and
    puts it in a easy to use dataframe, where each company has its own column.
    k is the number of the current_chunk"""
    # get a list of the company PERMNOs and get the number of dates
    permno = data_frame.PERMNO.unique().tolist()
    all_dates = data_frame.date.unique().tolist()
    nb_comp = len(permno) # Was used when we merged all the companies at once
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


def labelize_bsh(*args, threshold=0.02):
    """This is the simple function that puts a label on the future returns that we have.
    The initial implementation only gives buy sell or hold based on future returns"""
    future_returns_cols = [c for c in args]
    for future_returns_col in future_returns_cols:
        if future_returns_col > threshold:
            return 1
        if future_returns_col < -threshold:
            return -1
        return 0


def get_targets(df, nb_months=7):
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
        if (i % 320 == 0):
            print("{}%".format(round(i / len(permno) * 100)))
    tend = datetime.now() - tstart
    print("time: {}".format(tend.total_seconds()))
    return targets_df

def get_past_returns(df):
    '''This function will calculate the cumulative return of an asset over the past 11 months, but not including
    the current month's return, to better separate momentum from reverse momentum.
    The only input is the transformer return df, the output has the same shape.'''
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
        if isnan(col[0]): col[0] = 1
        for i in range(1,11):
            if isnan(col[i]):col[i] = 1
            cumcol[i] = prod(col[0:i])

        # After that we do on the 11 values before
        for i in range(11,len(col)):
            if isnan(col[i]):col[i] = 1
            cumcol[i] = prod(col[i-11:i])

        cumcol[0] = 1
        cumcol[:] = [x - 1 for x in cumcol]
        return cumcol

    # We now apply the to every column of the returns, excluding the dates
    df = df[:]
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(get_past_returns_col, axis=0)

    return df

def ml_dataframe(y_df, ret_df, permno_list, x1_df, x2_df, x3_df=0, x4_df=0):
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
        y_df_subtable = y_df_subtable.assign(PERMNO = [permno_list[i] for t in range(len(y_df_subtable))])
        # Merging
        y_df_subtable = y_df_subtable.merge(x1_df_subtable, how='left', on='date')
        y_df_subtable = y_df_subtable.merge(x2_df_subtable, how='left', on='date')
        # We append, but without the last value since there is no future price afterwards.
        # And without the first value since there is no TwelveToTwo past return
        ml_df = ml_df.append(y_df_subtable[1:-1], ignore_index=True, sort=False)

    # Deleting the first, useless value
    ml_df = ml_df.drop(ml_df.index[0])
        
    return ml_df


def large_ml_dataframe(y_df, ret_df, x1_df, x2_df, x3_df=0, x4_df=0, chunk_size=320):
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
    large_ml_df = pd.DataFrame({'date': [19509658], 'PERMNO': [00000], 'target': [0], 'price_1m': [0], 'price_122': [0]})
    
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
    mean = ml_dataframe.iloc[:,3:].mean()
    std = ml_dataframe.iloc[:,3:].std()
    ml_dataframe.iloc[:,3:] = (ml_dataframe.iloc[:,3:]-mean)/std

def fit_knn(ml_dataframe):

    print("FITTING KNN")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf.fit(X, y)

    outfile = open('pickle/KNN', 'wb')
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

    outfile = open('pickle/SVC', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))

def fit_rfc(ml_dataframe, max_depth=10):

    print("FITTING RFC")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']
    clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
    clf.fit(X, y)

    outfile = open('pickle/RFC', 'wb')
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

    outfile = open('pickle/LR', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    tend = datetime.now() - tstart
    print("Finished in {} seconds.".format(tend.total_seconds()))


def predict_knn(ml_dataframe):

    print("PREDICTING KNN")
    tstart = datetime.now()
    X = ml_dataframe[['price_1m', 'price_122']]
    y = ml_dataframe['target']

    infile = open('pickle/KNN', 'rb')
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

    infile = open('pickle/SVC', 'rb')
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

    infile = open('pickle/RFC', 'rb')
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

    infile = open('pickle/LR', 'rb')
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
            counter +=1

    print("Accuracy : {}".format(counter/len(agg_pred)))

def portfolio_performance(transformed_dataframe_returns, transformed_dataframe_mcap, prediction_transformed_dataframe):
    """This is the big function that will evaluate the performance of our investment strategy.
    From now on, it is very standardized, so everything will be done inside this one function.
    The inputs are simply the returns and market caps of the whole dataset, and the predicted choice of stocks."""

    # We start by replacing all the NaNs by 0, since not investing or not existing is essentially the same here
    prediction_transformed_dataframe[isnan(prediction_transformed_dataframe)] = 0
    # We now change all the values to int, for faster comparison
    prediction_transformed_dataframe = prediction_transformed_dataframe.astype("int64")

    # We initialize the weights matrices, and set to 1 all stocks that we invest in.
    # There are two portfolios here, one long and one short. We will subtract the returns later on.
    weights_long = prediction_transformed_dataframe.copy()
    weights_long[weights_long < 1] = 0

    weights_short = prediction_transformed_dataframe.copy()
    weights_short[weights_short > -1] = 0
    weights_short[weights_short == -1] = 1
    weights_short.date = weights_long.date # We reset the dates since they have disappeared in the previous operation.

    # There is one subtlety : investment decision is based on past info, so if the stock does
    # not exist anymore in the next month, then we will not be able to invest.

    # We shift the index of the returns by 2. One because we have no prediction for the first month
    # and one so as to shift the returns by 1 month and compare to investment decisions
    shifted_transformed_dataframe_returns = transformed_dataframe_returns.iloc[2:].copy()
    shifted_transformed_dataframe_returns = shifted_transformed_dataframe_returns.reset_index(drop=True)
    # We make sure that the returns df is of native dtype, to use the isnan function later
    shifted_transformed_dataframe_returns = shifted_transformed_dataframe_returns.astype("float64")
    # Note that shifted_... and weights_... have the same shape

    # We set to 0 investment decisions where there is no return in the next month.
    weights_long[isnan(shifted_transformed_dataframe_returns)] = 0
    weights_short[isnan(shifted_transformed_dataframe_returns)] = 0


    # We now need to find the respective weights of all stocks, using a value-weighted approach.
    # We calculate the relative weight of each chosen stock for each date

    # Taking absolute value because certain stocks have neg price & resetting index for division
    transformed_dataframe_mcap = transformed_dataframe_mcap.iloc[1:-1,1:].reset_index(drop=True).abs()
    # Setting to 0 all mcaps of stocks that are not selected
    transformed_dataframe_mcap_long = transformed_dataframe_mcap.copy()
    transformed_dataframe_mcap_short = transformed_dataframe_mcap.copy()
    transformed_dataframe_mcap_long[weights_long.iloc[:,1:] == 0] = 0
    transformed_dataframe_mcap_short[weights_short.iloc[:, 1:] == 0] = 0
    # Dividing
    weights_long.iloc[:,1:] = transformed_dataframe_mcap_long.divide(transformed_dataframe_mcap_long.sum(axis=1), axis='index')
    weights_short.iloc[:, 1:] = transformed_dataframe_mcap_short.divide(transformed_dataframe_mcap_short.sum(axis=1), axis='index')

    # We now want to calculate the performance of our two portfolios
    returns_df_long = shifted_transformed_dataframe_returns.iloc[:,1:].multiply(weights_long.iloc[:,1:], fill_value=0)
    returns_df_short = shifted_transformed_dataframe_returns.iloc[:,1:].multiply(weights_short.iloc[:, 1:], fill_value=0)

    # We compute the return for each date as the sum of the elements of that row
    performance_long = returns_df_long.sum(axis=1)
    performance_short = returns_df_short.sum(axis=1)

    performance = (performance_long-performance_short).mean()

    return performance



data = pd.read_csv("little_test_data.csv")
tdf_ret = transform_large_dataframe(data, chunk_size=2)
tdf_shrout = transform_large_dataframe(data, variable='SHROUT', chunk_size=2)
tdf_prc = transform_large_dataframe(data, variable='PRC', chunk_size=2)

# Obtaining the market caps

tdf_mcap = tdf_shrout.copy()
tdf_mcap.iloc[:,1:] = tdf_shrout.iloc[:,1:]*tdf_prc.iloc[:,1:]

targets_df = get_targets(tdf_ret)
cum_df = get_past_returns(tdf_ret)
cl = large_ml_dataframe(targets_df, tdf_ret, tdf_ret, cum_df, chunk_size=2)
standardize(cl)
fit_knn(cl)
knn = predict_knn(cl)
fit_svc(cl)
svc = predict_svc(cl)
fit_rfc(cl,20)
rfc = predict_rfc(cl)
fit_lr(cl)
lr = predict_lr(cl)

aggpred = aggregate_prediction(knn, svc, rfc, lr)
aggregate_accuracy(aggpred, cl)

pred_column = pd.DataFrame(aggpred, columns=['prediction'])
cl = cl.join(pred_column, sort=False)
new_tdf = transform_large_dataframe(cl, chunk_size=2, variable='prediction')

performance = portfolio_performance(tdf_ret, tdf_mcap, new_tdf)



# data = pd.read_csv("output_ret.csv", index_col=0)
# targets_df = get_targets(data)
# cum_df = get_past_returns(data)
# # targets_df = pd.read_csv("targets_df.csv", index_col=0)
# cl = large_ml_dataframe(targets_df, data, data, cum_df)
# cl.to_csv("cl.csv")
# cl = pd.read_csv("cl.csv")
# fit_ml(cl)
# predict_ml(cl)

# data = pd.read_csv("raw_data_full.csv")
# tdf = transform_large_dataframe(data, chunk_size=200)



