import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

file_path="/hdd8//dataset/coco/coco_LT_Missing_current/40/LT_train.json"
file_path2="/hdd8//dataset/coco/voc_LT_Missing_current/40/LT_train.json"

with open(file_path2) as f:
    file_content=json.load(f)

length=0
heat_map=np.zeros((20,20))
for i in range(20):
    length_coor=0
    # length_num=0
    vector=np.array([0]*20)
    for key in file_content.keys():
        if(file_content[key][i]==1):
            length_coor+=1
            vector+=np.array(file_content[key])
    heat_map[i]=vector/length_coor
    heat_map[i][i]=0


f, ax = plt.subplots(figsize=(12, 8))
x_axis=range(1,21)
# x_axis2=range(1,41,2)
y_axis=range(1,21)
file_content=pd.DataFrame(heat_map,columns=x_axis,index=y_axis)
# file_content=file_content.sort_values(by=[1],ascending=False)
ax=sns.heatmap(file_content, cmap="YlGnBu", rasterized=True)
plt.xlabel('Class Index',fontsize=20, color='k') #x轴label的文本和字体大小
plt.ylabel('Class Index',fontsize=20, color='k') #y轴label的文本和字体大小
plt.xticks(fontsize=13, rotation=270) #x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(fontsize=13) #y轴刻度的字体大小（文本包含在pd_data中了）
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
#设置colorbar的label文本和字体大小
# cbar = ax.collections[0].colorbar
# font = {'family': 'sans-serif',
#             'color': 'k',
#             'weight': 'normal',
#             'size': 20,}
# cbar.set_label(fontdict=font)
# plt.show()
plt.savefig("heatmap_4.pdf",format="pdf",dpi=600,bbox_inches='tight')

print(0)