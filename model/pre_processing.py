import pandas as pd
import numpy as np
import model.config as config

import sys
import gc


def compute_integer_stats(input_file, chunk_size):
    """
    统计训练集的整数特征项
    :param input_file: 输入文件路径
    :param chunk_size: 读取大文件时每次读取的行数
    :return:
    """

    # DataFrame是Python中Pandas库中的一种数据结构，
    # 它类似excel，是一种二维表。
    stats_integer = pd.DataFrame()

    clicks = 0
    impressions = 0
    pd.set_option('display.max_rows', None)
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0
    for chunk in reader:
        print("reading line:" + str(count * chunk_size))
        chunk_integer = chunk.iloc[:, 1:14]
        if count == 0:
            # 这里的数据结构：行是_2,_3等，列是max，min，count等
            stats_integer['max'] = chunk_integer.max()
            stats_integer['min'] = chunk_integer.min()
            stats_integer['sum'] = chunk_integer.sum()
            stats_integer['count'] = chunk_integer.count()
            print("count = 0:\n", stats_integer)
        else:

            stats_integer['max_chunk'] = chunk_integer.max()
            stats_integer['min_chunk'] = chunk_integer.min()
            stats_integer['sum_chunk'] = chunk_integer.sum()
            stats_integer['count_chunk'] = chunk_integer.count()

            stats_integer['max'] = stats_integer[['max', 'max_chunk']].max(axis=1)
            stats_integer['min'] = stats_integer[['min', 'min_chunk']].max(axis=1)
            stats_integer['sum'] = stats_integer[['sum', 'sum_chunk']].sum(axis=1)
            stats_integer['count'] = stats_integer[['count', 'count_chunk']].sum(axis=1)
            # 利用这种前后比对的方式就能求出每个特征的最大最小值了
            stats_integer.drop(['max_chunk', 'min_chunk', 'sum_chunk', 'count_chunk'], axis=1, inplace=True)
            print("count ！= 0:\n", stats_integer)
        clicks += chunk['_1'].sum()
        impressions += chunk.shape[0]
        print("inpressions:", impressions)
        print("count:", count)
        count += 1

    stats_integer['mean'] = stats_integer['sum'] / stats_integer['count']
    print(stats_integer['mean'])
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0

    for chunk in reader:
        print('Reading line:' + str(count * chunk_size))

        chunk_integer = chunk.iloc[:, 1:14]

        if count == 0:
            # 这里计算的是方差,利用前后比对的方式，就能求出整个文件每个元素的方差了
            stats_integer['sq_sum'] = ((chunk_integer - stats_integer['mean']) ** 2).sum()
        else:
            stats_integer['sq_sum_chunk'] = ((chunk_integer - stats_integer['mean']) ** 2).sum()
            stats_integer['sq_sum'] = stats_integer[['sq_sum', 'sq_sum_chunk']].sum(axis=1)
            stats_integer.drop(['sq_sum_chunk'], axis=1, inplace=True)

        count += 1

    stats_integer['std'] = (stats_integer['sq_sum'] / (stats_integer['count'] - 1)).apply(np.sqrt)
    stats_integer.drop(['sq_sum'], axis=1, inplace=True)

    print(stats_integer)
    stats_integer.to_csv(config.integer_stats_path)
    print("Total Clicks:" + str(clicks) + " Total Impressions:" + str(impressions))


def compute_category_stats(input_file, category_label, chunk_size):
    stats_category = pd.DataFrame()

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        print('Reading line:' + str(count * chunk_size))

        chunk_category = chunk.iloc[:, 14:]

        frame = pd.DataFrame()
        frame['category'] = chunk_category.groupby(category_label).size().index
        frame['count'] = chunk_category.groupby(category_label).size().values
        stats_category = pd.concat([stats_category, frame])

        # Aggregate on common category values
        frame = pd.DataFrame()
        frame['category'] = stats_category.groupby('category').sum().index
        frame['count'] = stats_category.groupby("category").sum().values
        stats_category = frame

        # Force garbage collection
        gc.collect()

        count += 1

    return stats_category.describe()


def compute_category_stats_all(input_file, chunk_size):
    stats_category = {}
    # 字典里面为每个列都建立一个DataFrame表
    for i in range(1, 27):
        stats_category['_' + str(i + 14)] = pd.DataFrame()

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        print('Reading line:' + str(count * chunk_size))

        chunk_category = chunk.iloc[:, 14:]

        for i in range(1, 27):
            category_label = '_' + str(i + 14)

            # 统计每一个标签对应的数量
            frame = pd.DataFrame()
            frame['category'] = chunk_category.groupby(category_label).size().index
            frame['count'] = chunk_category.groupby(category_label).size().values
            print("i:", i, ",frame1:\n", frame)
            stats_category[category_label] = pd.concat([stats_category[category_label], frame])

            # 每次把往期的和当前的块数据聚合起来，累计总和
            frame = pd.DataFrame()
            frame['category'] = stats_category[category_label].groupby('category').sum().index
            frame['count'] = stats_category[category_label].groupby("category").sum().values
            stats_category[category_label] = frame
            print("i:", i, ",frame2:\n", frame)

            gc.collect()

        count += 1

    stats_category_agg = pd.DataFrame()
    for i in range(1, 27):
        frame = stats_category['_' + str(i + 14)].groupby('category').sum().describe().transpose()
        frame.reset_index()
        frame.index = ['_' + str(i + 14)]
        stats_category_agg = pd.concat([stats_category_agg, frame])

    # print(stats_category_agg)
    # 最后的count表示一共有多少个标签，max表示出现次数最多的那个标签
    stats_category_agg.to_csv(config.category_stats_path)


if __name__ == "__main__":
    input_file = config.input_file_train
    chunk_size = 10000
    # compute_integer_stats(input_file, chunk_size)
    compute_category_stats_all(input_file, chunk_size)
