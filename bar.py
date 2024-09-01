import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from matplotlib import pyplot
from matplotlib import font_manager
# fontnamelist = font_manager.get()
# print(fontnamelist)
plt.rcParams['font.sans-serif'] = ['SimHei']
labels = ['Twitter15', 'Twitter16', ]
y1 = [0.823, 0.858, ]
y2 = [0.833, 0.865]
y3 = [0.826, 0.864]
y4 = [0.837, 0.874]
# plt.rcParams['font.family'] = ['SimHei']
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

x = np.arange(len(labels))
width = 0.15  # 每根柱子宽度

label_font = {
    'weight': 'bold',
    'size': 7.5,
    'family': 'SimHei'
}

# tick_params参数刻度线样式设置
# ax.tick_params(axis=‘x’, tickdir=‘in’, labelrotation=20)参数详解
# axis : 可选{‘x’, ‘y’, ‘both’} ，选择对哪个轴操作，默认是’both’
# which : 可选{‘major’, ‘minor’, ‘both’} 选择对主or副坐标轴进行操作
# direction/tickdir : 可选{‘in’, ‘out’, ‘inout’}刻度线的方向
# color : 刻度线的颜色，我一般用16进制字符串表示，eg：’#EE6363’
# width : float, 刻度线的宽度
# size/length : float, 刻度线的长度
# pad : float, 刻度线与刻度值之间的距离
# labelsize : float/str, 刻度值字体大小
# labelcolor : 刻度值颜色
# colors : 同时设置刻度线和刻度值的颜色
# bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=7.5, bottom=False)
# labelrotation=0 标签倾斜角度
ax.tick_params(axis='x', labelsize=7.5, bottom=False, labelrotation=0)

ax.set_xticks(x)
ax.set_ylim(ymin=0.8, ymax=0.89)
# 0 - 1800 ，200为一个间距
ax.set_yticks(np.arange(0.8, 0.89, 0.01))
ax.set_ylabel('(Acc)', fontdict=label_font)

ax.set_xticklabels(labels, fontdict=label_font,)
# ax.legend(markerscale=10,fontsize=12,prop=legend_font)
ax.legend(markerscale=10, fontsize=7.5)

'''
# 设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
'''
# 上下左右边框线宽
linewidth = 2
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_linewidth(linewidth)

    # Add some text for labels, title and custom x-axis tick labels, etc.


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
rects1 = ax.bar(x - 2*width, y1, width, label='text',ec='k',color='white',lw=.8,
               hatch='...')
rects2 = ax.bar(x  -width+ .05, y2, width, label='+KG',ec='k',color='white',
                lw=.8,hatch='///')

rects3 = ax.bar(x + width-width + .1, y3, width, label='+WK',ec='k',color='white',
                lw=.8,hatch='---')

rects4 = ax.bar(x + 2*width-width + .15, y4, width, label='+KG+WK',ec='k',color='white',
                lw=.8,hatch='xxx')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.legend(loc='upper right', frameon=True, fontsize=7.5)

fig.tight_layout()
plt.show()
# plt.savefig(r'C:\Users\Administrator\Desktop\p1.png', dpi=500)
