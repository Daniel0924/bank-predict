import pandas as pd
import model.config as config

a = [{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 100, 'c': 1}, {'a': 1, 'b': 100, 'c': 1}]
b = [{'a': 7, 'x': 8, 'y': 9}, {'a': 4, 'b': 5, 'c': 6}]


df = pd.read_csv(config.integer_stats_path)

print(df)

def Merge(dict1, dict2):
    dict2.update(dict1)
    return dict2


res = []
for (x, y) in zip(a, b):
    y.update(x)
    print("x", x)
    print("y", y)

    res.append(y)

print(res)

data_feature = [dict(x.items() + y.items()) for (x, y) in zip(a, b)]
