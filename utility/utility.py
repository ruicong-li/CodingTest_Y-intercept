import pandas as pd
import numpy as np

def compute_beta(ret, univ, window=252, mkt_index_method = 'equal', mkt_cap = None):
    if mkt_index_method == 'equal':
        benchmark_pnl = ret.mean(axis=1)
    elif mkt_index_method == 'mktcap':
        benchmark_weights = mkt_cap.rolling(window= 21, min_periods = 10).mean()
        benchmark_weights = benchmark_weights.divide(benchmark_weights.sum(axis=1, min_count=1), axis=0)
        benchmark_pnl = (benchmark_weights.shift() * ret).sum(axis=1, min_count=1)
    beta = pd.DataFrame(np.nan, index = ret.index, columns = ret.columns)
    for start_date, end_date in zip(univ.index[:-window], univ.index[window:]):
        valid_instr = ret.loc[start_date:end_date, :].count(axis=0) > (window//2)
        period_benchmark_ret = benchmark_pnl.loc[start_date:end_date].dropna()
        valid_days = period_benchmark_ret.index
        X = np.vstack([np.ones(len(period_benchmark_ret)), period_benchmark_ret.values]).T
        Y = ret.reindex(valid_days).loc[:, valid_instr].fillna(0).values
        beta.loc[end_date, valid_instr] = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)[1, :]
    return beta


def compute_weights(score, univ, method = 'UniformRank'):
    df_score = score.reindex_like(univ).mask(~univ)
    if method == 'UniformRank':
        weights = df_score.rank(axis=1, pct=True) - 0.5
    ## Rescale long leg and short leg to be sum of 1
    weights[weights > 0]= weights.divide(weights[weights > 0].sum(axis=1, min_count=1), axis=0)
    weights[weights < 0]= -weights.divide(weights[weights < 0].sum(axis=1, min_count=1), axis=0)
    return weights

def compute_summary(weights, ret, lags = [0]):
    df_summary = pd.DataFrame(index = lags)
    df_pnl = {}
    for lag in lags:
        df_weights = weights.shift(lag)
        pnl = (df_weights.shift() * ret).sum(axis=1, min_count=1)
        turnover = (df_weights - df_weights.shift()).abs().sum(axis=1)
        aum = df_weights.abs().sum(axis=1)
        df_summary.loc[lag, 'sharpe'] = pnl.mean() * np.sqrt(252) / pnl.std()
        df_summary.loc[lag, 'turnover'] = turnover.sum() / aum.sum()* 100
        df_pnl[lag] = pnl
    df_pnl = pd.concat(df_pnl, axis=1)
    return df_summary.transpose(), df_pnl