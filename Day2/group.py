from numpy import concatenate
import pandas as pd
import sys
concat_list = []

# print(len(sys.argv))
for i in range(1,len(sys.argv)):
    data = pd.read_csv('./Data/label_'+sys.argv[i]+'.csv')
    concat_list.append(data)

data = pd.concat(concat_list)
data.to_csv('./Data/Gesture_grouped_Data.csv', index=False)
