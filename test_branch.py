import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置 t 分布参数
df = 30  # 自由度
alpha = 0.05  # 显著性水平
t_critical = stats.t.ppf(1 - alpha / 2, df)  # 双侧检验的临界值

# t 分布的横坐标范围
x = np.linspace(-4, 4, 500)

# t 分布的概率密度函数
y = stats.t.pdf(x, df)

# 绘制 t 分布曲线
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='t-distribution (df=30)', color='b')

# 填充拒绝区域
plt.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.5, label=f'Rejection region (t > {t_critical:.2f})')
plt.fill_between(x, y, where=(x < -t_critical), color='red', alpha=0.5, label=f'Rejection region (t < {-t_critical:.2f})')

# 标记临界值
plt.axvline(t_critical, color='r', linestyle='--')
plt.axvline(-t_critical, color='r', linestyle='--')

# 添加文本说明
plt.text(t_critical + 0.2, 0.05, f'{t_critical:.2f}', color='r')
plt.text(-t_critical - 0.5, 0.05, f'{-t_critical:.2f}', color='r')

# 图表设置
plt.title("t-Distribution with Rejection Regions (df = 30)")
plt.xlabel('t values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# 显示图像
plt.show()
