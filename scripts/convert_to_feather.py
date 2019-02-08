import pandas as pd

target = [
    'historical_transactions',
    'new_merchant_transactions'
]
    # 'train',
    # 'test',

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    (pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
