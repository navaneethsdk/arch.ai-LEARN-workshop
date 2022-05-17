import cv2
import time
import pandas as pd
from hand_tracker import HandTracker
import sys
import numpy as np

try:
    label = sys.argv[1]
except:
    sys.exit("Please enter 1 label")

training_data = []
p_time = 0
c_time = 0
cap = cv2.VideoCapture(0)
detector = HandTracker()
finish = False
while not finish:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.get_raw_data()
    if len(lm_list) != 0:
        training_data.append(lm_list)
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Images", img)
    if cv2.waitKey(1)==27:
        finish=True

columns = []
for i in range(1, 43):
    columns.append('x_'+str(i))
    columns.append('y_'+str(i))
data = pd.DataFrame(training_data, columns=columns)
is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]
if len(rows_with_NaN.index) < len(data.index)-len(rows_with_NaN.index):
    data.dropna(inplace=True)
    data['label'] = label
    data.to_csv('Data/label_'+label+'.csv', index=False)
else:    
    rows_with_NaN = rows_with_NaN.replace(np.nan, 0)
    rows_with_NaN['label'] = label
    rows_with_NaN.to_csv('Data/label_'+label+'.csv', index=False)

print(f'Data Saved to CSV')