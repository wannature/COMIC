import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.signal

import random


shops = ["PLT-COCO", "PLT-VOC"]
sales_product_1 = [2962, 2569]
sales_product_2 = [5000, 4952]


hatches = ['---', '///', '\\\\\\', '...', 'xx', 'xxx']
# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

fig = plt.figure(figsize = (7,4))      
ax = fig.add_subplot(1, 1, 1) 
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15,
         }

ax.bar(xticks[0:2]+ 0.2 , sales_product_2[0:2], width=0.2,error_kw = {'ecolor' : '0.2', 'capsize' :8 },  color='skyblue', edgecolor='black',  linewidth=3,
                      hatch='.', label='Test Set')

ax.bar(xticks[0:2], sales_product_1[0:2], width=0.2, error_kw = {'ecolor' : '0.2', 'capsize' :8},  color='sandybrown', edgecolor='black', linewidth=3,
                      hatch='/', label='Train Set')

# ax.bar(xticks[0:5], sales_product_1[0:5], width=0.3, yerr = error_1[0:5],error_kw = {'ecolor' : '0.2', 'capsize' :8},  color='#ffff80', edgecolor='black', linewidth=3,
#                       hatch='/', label='Cake for Head Class')

# ax.bar(xticks[0:5] + 0.3, sales_product_2[0:5], width=0.3,yerr = error_2[0:5],error_kw = {'ecolor' : '0.2', 'capsize' :8 },  color='#ffcc66', edgecolor='black', linewidth=3,
#                       hatch='.', label='Cake (w/ bias) for Head Class')


# ax.bar(xticks[2:4], sales_product_1[2:4], width=0.3, yerr = error_1[2:4],error_kw = {'ecolor' : '0.2', 'capsize' :8 }, color='palegreen', edgecolor='black', linewidth=3,
#                      hatch='x', label='Cake Results  for Tail Class')
# ax.bar(xticks[2:4] + 0.3, sales_product_2[2:4], width=0.3,yerr = error_2[2:4],error_kw = {'ecolor' : '0.2', 'capsize' :8 }, color='#99ccff', edgecolor='black', linewidth=3,
#                       hatch='\\', label='Cake (w/ bias) for Tail Class')

ax.legend()

ax.set_xticks(xticks + 0.15)
ax.set_xticklabels(shops)

plt.ylim(0,6500)
pl.legend()
plt.legend(loc='upper center', ncol=2, fontsize=14)
plt.grid(True)
pl.xlabel(u'Dataset', fontsize=12)
pl.ylabel(u'Number', fontsize=12)
plt.savefig("dataset.png",dpi=600, bbox_inches='tight')

