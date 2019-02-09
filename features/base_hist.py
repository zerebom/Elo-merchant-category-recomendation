import argparse
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd

from utils.__init__ import timer


def get_arguments_hist():
    #パーサーインスタンスがを作成する
    parser = argparse.ArgumentParser()
    #overwriteする引数を与えている
    parser.add_argument(
        '--force', '-f', action='store_true', help='Overwrite existing files'
    )
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        #vがクラスであり、Featureのサブ（派生）クラスであり、
        #抽象クラスでないとき
        if inspect.isclass(v) and issubclass(v, Feature_hist) \
                and not inspect.isabstract(v):
            #vを返す
            yield v()


def generate_features_hist(namespace, overwrite):
    for f in get_features(namespace):
        #すでに、パスが存在する（特徴量ができた後なら飛ばす）
        if f.historical_transactions_path.exists() and f.new_merchant_transactions_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature_hist(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        #クラスの名前がすべて大文字なら
        if self.__class__.__name__.isupper():
            #小文字にします
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                "([A-Z])",
                lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip('_')

        self.historical_transactions = pd.DataFrame()
        self.new_merchant_transactions = pd.DataFrame()
        self.historical_transactions_path = Path(self.dir) / f'{self.name}_historical_transactions.feather'
        self.new_merchant_transactions_path = Path(self.dir) / f'{self.name}_new_merchant_transactions.feather'

    def run(self):
        with timer(self.name):
            self.create_features()
            self.historical_transactions=reduce_mem_usage(self.historical_transactions)
            self.new_merchant_transactions=reduce_mem_usage(self.new_merchant_transactions)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.historical_transactions.columns = prefix + self.historical_transactions.columns + suffix
            self.new_merchant_transactions.columns = prefix + self.new_merchant_transactions.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.historical_transactions.to_feather(str(self.historical_transactions_path))
        self.new_merchant_transactions.to_feather(str(self.new_merchant_transactions_path))

    def load(self):
        self.historical_transactions = pd.read_feather(str(self.historical_transactions_path))
        self.new_merchant_transactions = pd.read_feather(str(self.new_merchant_transactions_path))

    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

            end_mem = df.memory_usage().sum() / 1024**2
            print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
            print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

            return df
