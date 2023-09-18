import numpy as np


file_path="/hdd8//dataset/coco/voc_LT_Missing_current/40/distribution.txt"
file_content=np.loadtxt(file_path,delimiter=",",dtype=int)

print(np.sum(file_content))
print(np.max(file_content))
print(np.min(file_content))