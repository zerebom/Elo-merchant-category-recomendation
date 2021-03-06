import pandas as pd
import numpy as np
import re as re
import pandas as pd
import seaborn as sns
import time
from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Pclass(Feature):
    def create_features(self):
        self.train['Pclass'] = train['Pclass']
        self.test['Pclass'] = test['Pclass']


class Sex(Feature):
    def create_features(self):
        self.train['Sex'] = train['Sex'].replace(['male', 'female'], [0, 1])
        self.test['Sex'] = test['Sex'].replace(['male', 'female'], [0, 1])


class FamilySize(Feature):
    def create_features(self):
        self.train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        self.test['FamilySize'] = test['Parch'] + test['SibSp'] + 1


class Embarked(Feature):
    def create_features(self):
        self.train['Embarked'] = train['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)
        self.test['Embarked'] = test['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)


class Fare(Feature):
    def create_features(self):
        data = train.append(test)
        fare_mean = data['Fare'].mean()
        self.train['Fare'] = pd.qcut(
            train['Fare'].fillna(fare_mean),
            4,
            labels=False
        )
        self.test['Fare'] = pd.qcut(
            test['Fare'].fillna(fare_mean),
            4,
            labels=False
        )


class Age(Feature):
    def create_features(self):
        data = train.append(test)
        age_mean = data['Age'].mean()
        age_std = data['Age'].std()
        self.train['Age'] = pd.qcut(
            train['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )
        self.test['Age'] = pd.qcut(
            test['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )

class Quarter(Feature):
    def create_features(self):
        data=train.append(test)
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['quarter'] = df['first_active_month'].dt.quarter


def funcname(self, parameter_list):
    raise NotImplementedError


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
