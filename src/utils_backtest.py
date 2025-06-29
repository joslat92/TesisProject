def rolling_splits(df, train_len, test_len, stride):
    start = 0
    while start + train_len + test_len <= len(df):
        train_idx = slice(start, start + train_len)
        test_idx  = slice(start + train_len, start + train_len + test_len)
        yield df.iloc[train_idx], df.iloc[test_idx]
        start += stride
