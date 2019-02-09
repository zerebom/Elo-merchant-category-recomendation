import pandas as pd
import numpy as np
import re as re

from features.base_hist import Feature_hist, get_arguments_hist, generate_features_hist

Feature.dir = 'features'


class Pclass(Feature):
    def create_features(self):
        self.historical_transactions['Pclass'] = historical_transactions['Pclass']
        self.new_merchant_transactions['Pclass'] = new_merchant_transactions['Pclass']


class Sex(Feature):
    def create_features(self):
        self.historical_transactions['Sex'] = historical_transactions['Sex'].replace(['male', 'female'], [0, 1])
        self.new_merchant_transactions['Sex'] = new_merchant_transactions['Sex'].replace(['male', 'female'], [0, 1])


class FamilySize(Feature):
    def create_features(self):
        self.historical_transactions['FamilySize'] = historical_transactions['Parch'] + historical_transactions['SibSp'] + 1
        self.new_merchant_transactions['FamilySize'] = new_merchant_transactions['Parch'] + new_merchant_transactions['SibSp'] + 1


class Embarked(Feature):
    def create_features(self):
        self.historical_transactions['Embarked'] = historical_transactions['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)
        self.new_merchant_transactions['Embarked'] = new_merchant_transactions['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)


class Fare(Feature):
    def create_features(self):
        data = historical_transactions.append(new_merchant_transactions)
        fare_mean = data['Fare'].mean()
        self.historical_transactions['Fare'] = pd.qcut(
            historical_transactions['Fare'].fillna(fare_mean),
            4,
            labels=False
        )
        self.new_merchant_transactions['Fare'] = pd.qcut(
            new_merchant_transactions['Fare'].fillna(fare_mean),
            4,
            labels=False
        )


class Age(Feature):
    def create_features(self):
        data = historical_transactions.append(new_merchant_transactions)
        age_mean = data['Age'].mean()
        age_std = data['Age'].std()
        self.historical_transactions['Age'] = pd.qcut(
            historical_transactions['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )
        self.new_merchant_transactions['Age'] = pd.qcut(
            new_merchant_transactions['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


class Title(Feature):
    def create_features(self):
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        historical_transactions['Title'] = historical_transactions['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        historical_transactions['Title'] = historical_transactions['Name'].map(title_mapping).fillna(0)
        new_merchant_transactions['Title'] = new_merchant_transactions['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        new_merchant_transactions['Title'] = new_merchant_transactions['Title'].map(title_mapping).fillna(0)

        self.historical_transactions['Title'] = historical_transactions['Title']
        self.new_merchant_transactions['Title'] = new_merchant_transactions['Title']


if __name__ == '__main__':
    args = get_arguments_hist()

    historical_transactions = pd.read_feather('./data/input/historical_transactions.feather')
    new_merchant_transactions= pd.read_feather('./data/input/new_merchant_transactions.feather')

    generate_features_hist(globals(), args.force)
