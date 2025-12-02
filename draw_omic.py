import re
import matplotlib.pyplot as plt

# 定义log文件的路径
log_file_path = 'results/O_M_I_to_C_pre(img)_pgirm(v4-20)_normfc(False)_bsz(96)_rot(True)_smin(0.2)_tscl(0.9)_lr(0.005)_alpha(0.995)_scale(1)_floss(supcon)_flossw(0.1)_tmp(0.1)_seed(0)/log.txt'

# 初始化空列表来存储epoch和对应的loss值
epochs = []
losses = []

# 打开log文件并读取内容
with open(log_file_path, 'r') as file:
    for line in file:
        # 使用正则表达式匹配Train行的epoch和loss信息
        match = re.search(r'epoch:(\d+), Train: lr=\d+\.\d+, Loss=(\d+\.\d+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)

# 绘制损失随epoch的变化图
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o')
plt.title('Loss Variation with Epoch OMI_C')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 保存图像
plt.savefig('loss_variation_omic.png')

# 显示图像
plt.show()