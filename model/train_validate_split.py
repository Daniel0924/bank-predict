import model.config as config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 把数据集打乱顺序
# path = config.input_file_train
# data = pd.read_csv(path)
# data = shuffle(data)
# data.to_csv("Train_shuffle_400000.csv", index=False)

# TODO 将数据集分成40份，一份10000条数据
path = config.input_file_train_shuffle
data = pd.read_csv(path, chunksize=10000)
# TODO 保存数据集，命名格式：train_split000  train_split039
i = 0
for chunk in data:
    print("index:\n", chunk.index)
    chunk.columns = ['Id', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2',
                     'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
                     'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

    chunk.to_csv("/Users/daniel/PycharmProjects/bank-predict/data/train_split" + str(i).zfill(3) + ".csv", index=False)
    i = i + 1
    print("save:", i)

# 将特征划分到 X 中，标签划分到 Y 中
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(len(x))
print(len(y))
# 使用train_test_split函数划分数据集(训练集占75%，测试集占25%)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
