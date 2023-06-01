import os
import pandas as pd


file_list = []
for root, _dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".json"):
            file_list.append(os.path.join(root, file))

data = []
for file_path in file_list:
    df = pd.read_json(file_path)
    for column in df.columns:
        ser = df[column]
        ser = ser.dropna()
        source = ser.name
        for idx in ser.index:
            target = idx
            d = pd.DataFrame(ser[idx])
            d['source'] = source
            d['target'] = target
            data.append(d)
data = pd.concat(data)
new_columns = ['date', 'throughput', 'packetLoss', 'delay', 'jitter', 'availableBandwidth', 'layerIndex']
data = data.rename(columns={i: column for i, column in enumerate(new_columns)})
data['date'] = pd.to_datetime(data['date'], unit='s')
# never do sorting(because source, target mapping sequence is important)
# change data
available_layer = [0, 1, 2, 8, 9, 10, 16, 17, 18]
data = data[data['layerIndex'].isin(available_layer)]
print(data['layerIndex'].unique())
print(data)

data.to_csv('structed_data.csv', sep=',')


#                               date  throughput  packetLoss     delay    jitter  availableBandwidth  layerIndex    source    target
# 0    2023-05-30 06:28:48.091298048    0.000000         0.0  0.000000  1.000000            5.000000           0  d9654776  14161e8e
# 1    2023-05-30 06:28:48.303307776    0.000000         0.0  0.000000  1.000000            5.000000           1  d9654776  14161e8e
# 2    2023-05-30 06:28:48.466922752    0.416880         0.0  0.000000  1.000000            5.000000           1  d9654776  14161e8e
# 3    2023-05-30 06:28:48.525847808    0.416880         0.0  0.000000  1.000000            5.000000           0  d9654776  14161e8e
# 4    2023-05-30 06:28:48.592539648    0.606526         0.0  0.000000  1.000000            5.000000           8  d9654776  14161e8e
# ...                            ...         ...         ...       ...       ...                 ...         ...       ...       ...
# 5398 2023-05-31 02:31:56.126579456    0.000000         0.0  0.052454  6.897713            0.290517           1  3b969241  7f2fefdd
# 5399 2023-05-31 02:31:57.136227328    0.000000         0.0  0.052454  6.897713            0.290517           1  3b969241  7f2fefdd
# 5400 2023-05-31 02:31:58.140236800    0.000000         0.0  0.052454  6.897713            0.290517           1  3b969241  7f2fefdd
# 5401 2023-05-31 02:31:59.143382016    0.000000         0.0  0.052454  6.897713            0.290517           1  3b969241  7f2fefdd
# 5402 2023-05-31 02:32:00.152213760    0.000000         0.0  0.052454  6.897713            0.290517           1  3b969241  7f2fefdd