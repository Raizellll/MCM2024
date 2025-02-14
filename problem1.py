'''问题一：不同情况下抽样检测最小样本量的数学模型'''

import scipy.stats as stats
import math

def sample_size(defective_rate, effect_level ,confidence_level):
    # 计算样本大小
    z = abs(stats.norm.ppf((1 - confidence_level)))  # z值
    p = defective_rate  # 次品率
    n = (z**2 * p * (1 - p)) / (effect_level**2)  # 抽样公式
    print(z)
    return math.ceil(n)  # 向上取整
    
# 问题1 (1)
n1 = sample_size(0.1, 0.02,0.95)  # 95%信度
print(f'在95%信度下的抽样次数: {n1}')
# 问题1 (2)
n2 = sample_size(0.1,0.02 ,0.90)  # 90%信度
print(f'在90%信度下的抽样次数: {n2}')
