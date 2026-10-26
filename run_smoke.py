"""首次验证：一个数据集、一个种子、完整数据、中位数 + Isolation Forest。"""
from copy import deepcopy
from research.config import DEFAULT
from research.experiment import run

if __name__ == '__main__':
    config = deepcopy(DEFAULT)
    config.update(seeds=[42], missing_rates=[0.0], imputers=['median'], detectors=['iforest'])
    print(run(config, label='smoke'))
