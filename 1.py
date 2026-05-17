import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色定义
colors = {
    'input': '#1f77b4',      # 蓝色
    'process': '#2ca02c',    # 绿色
    'output': '#d62728',     # 红色
    'loss': '#ff7f0e',       # 橙色
    'connection': '#7f7f7f'  # 灰色
}

# 1. 输入部分
# 演示图和测试图
ax.text(1, 8.5, '演示对齐 Ademo', ha='center', va='center', fontsize=12, color=colors['input'])
ax.add_patch(patches.Rectangle((0.3, 8), 1.4, 0.8, linewidth=1, edgecolor=colors['input'], facecolor='none'))
ax.text(1, 7.5, '+', ha='center', va='center', fontsize=16)

ax.text(1, 6.5, '测试对齐 Atest', ha='center', va='center', fontsize=12, color=colors['input'])
ax.add_patch(patches.Rectangle((0.3, 6), 1.4, 0.8, linewidth=1, edgecolor=colors['input'], facecolor='none'))

# 负样本
ax.text(1, 4.5, '负样本 Atest_j', ha='center', va='center', fontsize=12, color=colors['input'])
ax.add_patch(patches.Rectangle((0.3, 4), 1.4, 0.8, linewidth=1, edgecolor=colors['input'], facecolor='none'))
ax.text(1, 3.5, '...', ha='center', va='center', fontsize=16)
ax.add_patch(patches.Rectangle((0.3, 2.5), 1.4, 0.8, linewidth=1, edgecolor=colors['input'], facecolor='none'))

# 2. 异构图构建
ax.text(4, 8.5, '异构图构建', ha='center', va='center', fontsize=14, color=colors['process'], weight='bold')
ax.add_patch(patches.Rectangle((3, 7.5), 2, 2, linewidth=2, edgecolor=colors['process'], facecolor='#e8f5e9'))

# 图例说明
ax.text(4, 7, '• 内部边\n• 跨物体边\n• 演示-测试边\n• 位置编码', ha='center', va='center', fontsize=10)

# 3. Graph Transformer 处理
ax.text(7, 8.5, 'Graph Transformer\n(4层, 4头)', ha='center', va='center', fontsize=14, color=colors['process'], weight='bold')
ax.add_patch(patches.Rectangle((6, 7.5), 2, 2, linewidth=2, edgecolor=colors['process'], facecolor='#e8f5e9'))

# 层数表示
for i in range(4):
    y_pos = 7.7 + i * 0.4
    ax.add_patch(patches.Rectangle((6.2, y_pos), 1.6, 0.3, linewidth=1, edgecolor='gray', facecolor='white'))
    ax.text(7, y_pos+0.15, f'Layer {i+1}', ha='center', va='center', fontsize=8)

# 4. Class Token 输出
ax.text(10, 8.5, 'Class Token\n特征向量', ha='center', va='center', fontsize=12, color=colors['process'])
ax.add_patch(patches.Circle((10, 8), 0.3, linewidth=2, edgecolor=colors['process'], facecolor='#e8f5e9'))

# 5. MLP 输出能量
ax.text(12.5, 8.5, 'MLP\n(2层, 256维)', ha='center', va='center', fontsize=14, color=colors['process'], weight='bold')
ax.add_patch(patches.Rectangle((11.5, 7.5), 2, 2, linewidth=2, edgecolor=colors['process'], facecolor='#e8f5e9'))

# 6. 能量输出
ax.text(12.5, 6, '能量值 E', ha='center', va='center', fontsize=14, color=colors['output'], weight='bold')
ax.add_patch(patches.Rectangle((11.5, 5.5), 2, 1, linewidth=2, edgecolor=colors['output'], facecolor='#ffebee'))

# 7. 概率转换
ax.text(12.5, 4, '概率转换\nexp(-E)/Z', ha='center', va='center', fontsize=12, color=colors['process'])
ax.add_patch(patches.Rectangle((11.5, 3.5), 2, 1, linewidth=2, edgecolor=colors['process'], facecolor='#fff3e0'))

# 8. InfoNCE 损失计算
ax.text(12.5, 2, 'InfoNCE 损失', ha='center', va='center', fontsize=14, color=colors['loss'], weight='bold')
ax.add_patch(patches.Rectangle((11.5, 1.5), 2, 1, linewidth=2, edgecolor=colors['loss'], facecolor='#ffe0b2'))

# 连接线
# 输入到异构图
ax.annotate('', xy=(2, 8.4), xytext=(3, 8.4), arrowprops=dict(arrowstyle='->', color=colors['connection']))
ax.annotate('', xy=(2, 6.4), xytext=(3, 6.4), arrowprops=dict(arrowstyle='->', color=colors['connection']))
ax.annotate('', xy=(2, 4.4), xytext=(3, 4.4), arrowprops=dict(arrowstyle='->', color=colors['connection']))
ax.annotate('', xy=(2, 2.9), xytext=(3, 2.9), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# 异构图到 Graph Transformer
ax.annotate('', xy=(5, 8.5), xytext=(6, 8.5), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# Graph Transformer 到 Class Token
ax.annotate('', xy=(8, 8.5), xytext=(9.7, 8.5), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# Class Token 到 MLP
ax.annotate('', xy=(10.3, 8), xytext=(11.5, 8), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# MLP 到能量
ax.annotate('', xy=(12.5, 7.5), xytext=(12.5, 6.5), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# 能量到概率
ax.annotate('', xy=(12.5, 5.5), xytext=(12.5, 4.5), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# 概率到损失
ax.annotate('', xy=(12.5, 3.5), xytext=(12.5, 2.5), arrowprops=dict(arrowstyle='->', color=colors['connection']))

# 返回到 Graph Transformer (梯度反向传播)
ax.annotate('', xy=(11.5, 2), xytext=(8, 2), arrowprops=dict(arrowstyle='->', color=colors['connection']))
ax.annotate('', xy=(8, 2), xytext=(4, 2), arrowprops=dict(arrowstyle='->', color=colors['connection']))
ax.annotate('', xy=(4, 2), xytext=(2, 2), arrowprops=dict(arrowstyle='->', color=colors['connection'], linestyle='dashed'))
ax.text(3, 2.2, '梯度反向传播', ha='center', va='center', fontsize=10, color='gray', style='italic')

# 标题
plt.title('IGA 能量模型架构图', fontsize=16, pad=20)

# 保存图片
energy_model_architecture = 'iga_energy_model_architecture.png'
plt.tight_layout()
plt.savefig(energy_model_architecture, dpi=300, bbox_inches='tight')
print(energy_model_architecture)