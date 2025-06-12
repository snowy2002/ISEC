import os
import matplotlib.pyplot as plt


def plot_learning_curve(episodes, records, title, ylabel, figure_file):# 绘制学习曲线
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, dpi = 400)
    plt.show()


def create_directory(path: str, sub_dirs: list):# 创建下属文件夹
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')

