"""PyCharm 打开本文件，选择项目 .venv 解释器，点击运行。无需命令行参数。"""
from research.experiment import run

if __name__ == '__main__':
    result = run()
    print(f'实验完成，结果文件夹：{result}')
