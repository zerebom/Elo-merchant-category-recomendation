Elo-merchant-category-recomendation
===
- [Kaggle Titanic](https://www.kaggle.com/c/titanic) example of my own, inspired by [flowlight0's repo](https://github.com/flowlight0/talkingdata-adtracking-fraud-detection).
- You can get the score = 0.76555 at the version of 2018-12-28.
- Japanese article can be seen [here](https://upura.hatenablog.com/entry/2018/12/28/225234).

# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── features(書く特徴量の保存)
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models（学習機。入力:df,params 出力:予測結果）
│   └── lgbm.py
├── notebooks
│   └── eda.ipynb
├── scripts(汎用コード)
│   └── convert_to_feather.py
├── utils(汎用コード)
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py(コンペに応じて書く、計算コード)
└── tox.ini
```
# Commands

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

## flake8

```
flake8 .
```
