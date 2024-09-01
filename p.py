# import scipy.sparse as sp
#
# a = [6, 5, 3, 5, 3, 6, 2, 6, 2, 3, 4, 3, 4, 2, 1, 2, 1, 4, 7, 4, 7, 1, 0, 1, 0, 7]
# b = [5, 6, 5, 3, 6, 3, 6, 2, 3, 2, 3, 4, 2, 4, 2, 1, 4, 1, 4, 7, 1, 7, 1, 0, 7, 0]
# w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#
# # print((len(new_edgeindex[0]), len(new_edgeindex[1])))
#
# new_edgeindex = sp.csr_matrix((w, (a, b)),shape=(max(a)+1, max(b)+1))
# print(new_edgeindex)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
#
# 输入数据
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y1 = [0.868, 0.874, 0.874, 0.871, 0.871, 0.875, 0.870,0.868,\
      0.867,0.869,0.869]
y2 = [0.820, 0.824, 0.823, 0.829, 0.837, 0.834, 0.833,\
      0.834,0.834,0.833,0.833]

# 设置颜色代码

# color1 = "#038355" # 孔雀绿
# color2 = "#ffc34e" # 向日黄
# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 7.5}
plt.rc('font', **font)
fig,ax = plt.subplots()
# 绘图
sns.set_style("whitegrid") # 设置背景样式
#color=color1,
sns.lineplot(x=x, y=y1, color='black', linewidth=1.0, marker="o", markersize=7.5, markeredgecolor="white", markeredgewidth=1.5, label='RDLNP', linestyle='--')
sns.lineplot(x=x, y=y2 ,color='black', linewidth=1.0, marker="s", markersize=7.5, markeredgecolor="white", markeredgewidth=1.5, label='RVNN')

# 添加标题和标签
plt.title("Title", fontweight='bold', fontsize=7.5)
plt.xlabel("Detection deadline(mins)", fontsize=7.5)
plt.ylabel("Accuracy(%)", fontsize=7.5)

# 添加图例
plt.legend(loc='lower right', frameon=True, fontsize=7.5)
plt.grid(False)
# 设置刻度字体和范围
plt.xticks(fontsize=7.5)
plt.yticks(fontsize=7.5)
plt.xlim(-0.1, 1.1)
xx = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.xticks(xx)
ax.set_ylim(ymin=0.8, ymax=0.89)

# plt.xlim(-1,)

# ax.xaxis.set_major_locator(ticker.FixedLocator(x))
# yy = [0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90]
# y_tick = yy
# ax.yaxis.set_major_locator(ticker.FixedLocator(y_tick))
# ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_tick))

# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',-1))



# plt.ylim(0.5,1)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

plt.savefig('lineplot.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.show()
#
# import matplotlib.pyplot as plt
#
# x_values = list(range(11))
# # x轴的数字是0到10这11个整数
# y_values = [x ** 2 for x in x_values]
# # y轴的数字是x轴数字的平方
# plt.plot(x_values, y_values, c='green')
# # 用plot函数绘制折线图，线条颜色设置为绿色
# plt.title('Squares', fontsize=24)
# # 设置图表标题和标题字号
# plt.tick_params(axis='both', which='major', labelsize=14)
# # 设置刻度的字号
# plt.xlabel('Numbers', fontsize=14)
# # 设置x轴标签及其字号
# plt.ylabel('Squares', fontsize=14)
# # 设置y轴标签及其字号
# plt.show()


