'''问题四(2)：模糊次品率下的生产过程决策'''

import numpy as np
import pandas as pd
import deap

def sale(N, S):
    # N: 客户需求
    # S: 售价
    return N * S

def check_last(X3, N, w, I3):
    # X3: 是否检测成品
    # N: 客户需求
    # w: 成品最终良品率
    # I3: 成品检测费
    return X3 * N * I3 / w

def replace_last(X3, N, w, R):
    # X3: 是否检测成品
    # N: 客户需求
    # w: 成品最终良品率 omega
    # R: 调换费

    # 只换你,等比数列全部项求和得到的结果
    return (1 - X3) * (N / w) * ((1 - w) / w) * R

def check_1(X1, N, w, I1, Mu):
    # X1: 是否检测零件1
    # N: 客户需求
    # w: 成品最终良品率
    # I1: 零件1检测费
    # u: 零件1良品率 mu
    return N * X1 * I1 / w / Mu

def check_2(X2, N, w, I2, Lambda):
    # X2: 是否检测零件2
    # N: 客户需求
    # w: 成品最终良品率
    # I2: 零件2检测费
    # l: 零件2良品率 lambda
    return N * X2 * I2 / w / Lambda

def disassemble(X4, N, w, D):
    # N: 客户需求
    # w: 成品最终良品率
    # D: 拆解价格
    return N * X4 * (1 / w - 1) * D

def purchase_1(a, P1):
    # a: 零件1购入数量
    # P1: 零件1的成本
    return a * P1

def purchase_2(b, P2):
    # b: 零件2购入数量
    # P2: 零件2的成本
    return b * P2

def assemble(P3, N, w):
    # P3: 成品的组装费
    # N: 客户需求
    # w: 成品最终良品率
    return P3 * N / w

def profit_inner(N, X1, X2, X3, X4, I1, I2, I3, P1, P2, P3, S, R, D, w, Mu, Lambda, a, b):
    return sale(N, S) - check_last(X3, N, w, I3) - replace_last(X3, N, w, R) - \
    check_1(X1, N, w, I1, Mu) - check_2(X2, N, w, I2, Lambda) - disassemble(X4, N, w, D) - \
    purchase_1(a, P1) - purchase_2(b, P2) - assemble(P3, N, w)

def get_ab(N, w, Mu, X1, X2, X4): # 100, 0.72, 0.9, 1, 0, 1 → 
    '''
    return a, b
    获得 - 回收
    '''

    def get_gain_recycle(X):
        '''
        return gain, recycle
        X: X1/X2, 对于零件1/2是否检测
        '''
        # 获得数量
        if X == 0:
            gain = N / w
        else: # X == 1
            gain = N / w / Mu
        # 回收数量
        recycle = X4 * N * (1 / w - 1)
        return gain, recycle

    gain1, recycle1 = get_gain_recycle(X1)
    gain2, recycle2 = get_gain_recycle(X2)

    # 因为题设是畅销品,设N为很大的值,因此上取整的影响足够小,可以忽略不计
    
    # import math
    # a = math.ceil(int((gain1 - recycle1) * 100) / 100)
    # b = math.ceil(int((gain2 - recycle2) * 100) / 100) # 先截断到小数第三位,再向上取整,防止小数bug
    
    
    a = gain1 - recycle1
    b = gain2 - recycle2
    return a, b

def get_w(u, l, Omega):
    # u: 零件1良品率
    # l: 零件2良品率
    # Omega: 两良好零件下合成成功率
    return u * l * Omega

def get_u(Mu, X1):
    # Mu: 零件1的购买良品率
    # X1: 是否检测零件1
    if X1 == 0:
        return Mu
    elif X1 == 1:
        return 1

def get_l(Lambda, X2):
    # Lambda: 零件2的购买良品率
    # X2: 是否检测零件2
    if X2 == 0:
        return Lambda
    elif X2 == 1:
        return 1
    
def profit(N, X, I, P, S, R, D, Mu, Lambda, Omega):
    '''
    N: 客户需求
    X = (X1, X2, X3, X4): 是否检测零件1, 2, 成品, 是否拆解成品
    I = (I1, I2, I3): 零件1, 2, 成品检测费
    P = (P1, P2, P3): 零件1, 2的成本, 成品组装费
    S: 售价
    R: 调换费
    D: 拆解费
    Mu: 零件1的购买良品率
    Lambda: 零件2的购买良品率
    Omega: 两良好零件下合成成功率
    '''
    X1, X2, X3, X4 = X
    I1, I2, I3 = I
    P1, P2, P3 = P
    u = get_u(Mu, X1)
    l = get_l(Lambda, X2)
    w = get_w(u, l, Omega)
    a, b = get_ab(N, w, Mu, X1, X2, X4)
    return profit_inner(N, X1, X2, X3, X4, I1, I2, I3, P1, P2, P3, S, R, D, w, Mu, Lambda, a, b)


# 生产相关数据封装
def get_production_data():
    N = 100  # 客户需求量
    I = (2, 3, 3)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 6   # 调换费
    D = 5   # 拆解费

    # 中间值封装
    Mu_middle = 0.8  # 零件1的中间良品率
    Lambda_middle = 0.8  # 零件2的中间良品率
    Omega_middle = 0.8  # 成品合成成功率的中间值

    # 返回所有封装数据
    return N, I, P, S, R, D, Mu_middle, Lambda_middle, Omega_middle

# 定义三角模糊数
class TriangularFuzzyNumber:
    def __init__(self, l, m, u):
        self.l = l  # 最小值
        self.m = m  # 最可能值
        self.u = u  # 最大值

    # 解模糊化方法
    def defuzzify(self, method="centroid", q=0.5):
        if method == "centroid":
            return (self.l + self.m + self.u) / 3
        elif method == "optimism-pessimism":
            return q * self.u + (1 - q) * self.l
        else:
            raise ValueError("Unknown defuzzification method")

# 使用模糊数的函数
def get_fuzzy_numbers(Mu_middle, Lambda_middle, Omega_middle):
    # 根据中间值计算模糊数左右值（±0.02）
    Mu_fuzzy = TriangularFuzzyNumber(Mu_middle - 0.02, Mu_middle, Mu_middle + 0.02)  # 零件1的模糊良品率
    Lambda_fuzzy = TriangularFuzzyNumber(Lambda_middle - 0.02, Lambda_middle, Lambda_middle + 0.02)  # 零件2的模糊良品率
    Omega_fuzzy = TriangularFuzzyNumber(Omega_middle - 0.02, Omega_middle, Omega_middle + 0.02)  # 成品合成成功率的模糊数

    # 解模糊化
    Mu = Mu_fuzzy.defuzzify(method="centroid")
    Lambda = Lambda_fuzzy.defuzzify(method="centroid")
    Omega = Omega_fuzzy.defuzzify(method="centroid")

    return Mu, Lambda, Omega

from deap import base, creator, tools
import random

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册适应度评估函数
# 注册适应度评估函数
def evaluate(individual):
    # 解包 9 个值：N, I, P, S, R, D, Mu_middle, Lambda_middle, Omega_middle
    N, I, P, S, R, D, Mu_middle, Lambda_middle, Omega_middle = get_production_data()
    
    # 使用中间值计算模糊数并解模糊化
    Mu, Lambda, Omega = get_fuzzy_numbers(Mu_middle, Lambda_middle, Omega_middle)
    
    X = individual
    return profit(N, X, I, P, S, R, D, Mu, Lambda, Omega),

# 其余逻辑保持不变


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt


# 五：绘制适应度曲线
def plot_fitness_curve(logbook_fixed, logbook_adaptive):
    generations_fixed = logbook_fixed.select("gen")
    avg_fitness_fixed = logbook_fixed.select("avg")
    max_fitness_fixed = logbook_fixed.select("max")
    
    generations_adaptive = logbook_adaptive.select("gen")
    avg_fitness_adaptive = logbook_adaptive.select("avg")
    max_fitness_adaptive = logbook_adaptive.select("max")
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations_fixed, avg_fitness_fixed, label='Average Fitness (Fixed)', color='blue', linestyle='--')
    plt.plot(generations_fixed, max_fitness_fixed, label='Max Fitness (Fixed)', color='blue', linestyle='-')
    plt.plot(generations_adaptive, avg_fitness_adaptive, label='Average Fitness (Adaptive)', color='green', linestyle='--')
    plt.plot(generations_adaptive, max_fitness_adaptive, label='Max Fitness (Adaptive)', color='green', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Curves (Fixed vs Adaptive Parameters)')
    plt.legend()
    plt.show()

# 六：固定参数的遗传算法
def run_fixed_parameters():
    population = toolbox.population(n=100)
    generations = 50

    logbook_fixed = tools.Logbook()
    logbook_fixed.header = ['gen', 'nevals', 'avg', 'std', 'min', 'max']

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(generations):
        cxpb, mutpb = 0.5, 0.2  # 固定的交叉和变异概率
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)
        record = stats.compile(population)
        logbook_fixed.record(gen=gen, nevals=len(population), **record)
    
    return logbook_fixed, population

# 七：自适应参数的遗传算法
def adaptive_params(gen, max_gen):
    initial_cxpb = 0.8
    final_cxpb = 0.5
    initial_mutpb = 0.3
    final_mutpb = 0.1

    cxpb = initial_cxpb - (initial_cxpb - final_cxpb) * (gen / max_gen)
    mutpb = initial_mutpb - (initial_mutpb - final_mutpb) * (gen / max_gen)
    return cxpb, mutpb

def run_adaptive_parameters():
    population = toolbox.population(n=100)
    generations = 50

    logbook_adaptive = tools.Logbook()
    logbook_adaptive.header = ['gen', 'nevals', 'avg', 'std', 'min', 'max']

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(generations):
        cxpb, mutpb = adaptive_params(gen, generations)
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)
        record = stats.compile(population)
        logbook_adaptive.record(gen=gen, nevals=len(population), **record)
    
    return logbook_adaptive, population
# 主函数
if __name__ == "__main__":
    # 运行固定参数的遗传算法
    logbook_fixed, pop_fixed = run_fixed_parameters()
    
    # 运行自适应参数的遗传算法
    logbook_adaptive, pop_adaptive = run_adaptive_parameters()
    
    # 绘制两条曲线
    plot_fitness_curve(logbook_fixed, logbook_adaptive)

    # 输出最优个体和其对应的利润
    best_individual_fixed = tools.selBest(pop_fixed, 1)[0]
    best_individual_adaptive = tools.selBest(pop_adaptive, 1)[0]
    
    print(f"Best individual (Fixed Parameters): {best_individual_fixed}")
    print(f"Profit (Fixed Parameters): {evaluate(best_individual_fixed)[0]}")

    print(f"Best individual (Adaptive Parameters): {best_individual_adaptive}")
    print(f"Profit (Adaptive Parameters): {evaluate(best_individual_adaptive)[0]}")

# 运行固定参数的遗传算法，接受模糊数作为参数
def run_fixed_parameters_with_fuzzy(Mu, Lambda, Omega):
    population = toolbox.population(n=100)
    generations = 50

    logbook_fixed = tools.Logbook()
    logbook_fixed.header = ['gen', 'nevals', 'avg', 'std', 'min', 'max']

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(generations):
        cxpb, mutpb = 0.5, 0.2  # 固定的交叉和变异概率
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)
        record = stats.compile(population)
        logbook_fixed.record(gen=gen, nevals=len(population), **record)
    
    return logbook_fixed, population

def calculate_optimal_decisions_for_different_q():
    # q的取值范围从0.1到0.9，步长为0.1
    q_values = np.arange(0.1, 1.0, 0.1)
    
    # 存储不同q值下的最优决策
    q_optimal_decisions = []
    
    for q in q_values:
        # 获取模糊数并根据当前q值进行解模糊化
        Mu_fuzzy = TriangularFuzzyNumber(0.7, 0.8, 0.9)
        Lambda_fuzzy = TriangularFuzzyNumber(0.7, 0.8, 0.9)
        Omega_fuzzy = TriangularFuzzyNumber(0.7, 0.8, 0.9)
        
        Mu = Mu_fuzzy.defuzzify(method="optimism-pessimism", q=q)
        Lambda = Lambda_fuzzy.defuzzify(method="optimism-pessimism", q=q)
        Omega = Omega_fuzzy.defuzzify(method="optimism-pessimism", q=q)

        # 运行遗传算法，得到最优个体（决策）
        logbook_fixed, pop_fixed = run_fixed_parameters_with_fuzzy(Mu, Lambda, Omega)
        best_individual_fixed = tools.selBest(pop_fixed, 1)[0]
        
        # 将q值和对应的最优个体（决策）保存
        q_optimal_decisions.append((q, best_individual_fixed))

    # 返回结果
    return q_optimal_decisions



if __name__ == "__main__":
    # 计算不同q值对应的最优决策
    q_optimal_decisions = calculate_optimal_decisions_for_different_q()

    # 输出q值对应的最优决策
    print("q值和对应的最优决策：")
    for q, decision in q_optimal_decisions:
        print(f"q={q:.1f}, 最优决策={decision}")

def get_scenario_1_data():
    N = 100  # 客户需求量
    I = (2, 3, 3)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 6   # 调换费
    D = 5   # 拆解费
    Mu = 0.9  # 零件1的购买良品率
    Lambda = 0.9  # 零件2的购买良品率
    Omega = 0.9  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

def get_scenario_2_data():
    N = 100  # 客户需求量
    I = (2, 3, 3)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 6   # 调换费
    D = 5   # 拆解费
    Mu = 0.8  # 零件1的购买良品率
    Lambda = 0.8  # 零件2的购买良品率
    Omega = 0.8  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

def get_scenario_3_data():
    N = 100  # 客户需求量
    I = (2, 3, 3)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 30   # 调换费
    D = 5   # 拆解费
    Mu = 0.9  # 零件1的购买良品率
    Lambda = 0.9  # 零件2的购买良品率
    Omega = 0.9  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

def get_scenario_4_data():
    N = 100  # 客户需求量
    I = (1, 1, 2)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 30   # 调换费
    D = 5   # 拆解费
    Mu = 0.8  # 零件1的购买良品率
    Lambda = 0.8  # 零件2的购买良品率
    Omega = 0.8  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

def get_scenario_5_data():
    N = 100  # 客户需求量
    I = (8, 1, 2)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 10   # 调换费
    D = 5   # 拆解费
    Mu = 0.9  # 零件1的购买良品率
    Lambda = 0.8  # 零件2的购买良品率
    Omega = 0.9  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

def get_scenario_6_data():
    N = 100  # 客户需求量
    I = (2, 3, 3)  # 零件1、2，成品的检测成本
    P = (4, 18, 6)  # 零件1、2，成品的组装成本
    S = 56  # 售价
    R = 10   # 调换费
    D = 40   # 拆解费
    Mu = 0.95  # 零件1的购买良品率
    Lambda = 0.95  # 零件2的购买良品率
    Omega = 0.95  # 两良好零件下合成成功率
    return N, I, P, S, R, D, Mu, Lambda, Omega

decisions = []

get_production_data = get_scenario_1_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

# 输出q值对应的最优决策
print("q值和对应的最优决策：")

p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

'''注释: 这种情况也是收敛的，只是对于题设的参数及本论文做出的假设，
恰好有两种情况在数值上完全相同。
如果考虑实际的零件1良品率会偏大的情况，可以排除其中一者，获得唯一的最优决策。
因此这里只加入零件1良品率会更大的其中一者。
'''

get_production_data = get_scenario_2_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

# 输出q值对应的最优决策
print("q值和对应的最优决策：")

p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

get_production_data = get_scenario_3_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

# 输出q值对应的最优决策
print("q值和对应的最优决策：")
p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

get_production_data = get_scenario_4_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

get_production_data = get_scenario_5_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

get_production_data = get_scenario_6_data
# 计算不同q值对应的最优决策
q_optimal_decisions = calculate_optimal_decisions_for_different_q()

p_decisions = []
for q, decision in q_optimal_decisions:
    print(f"q={q:.1f}, 最优决策={decision}")
    if decision not in p_decisions:
        p_decisions.append(decision)

decisions.append(p_decisions[0])

# 情况1-6的题设数值

# 检测成本
I = [(2, 3, 3), 
     (2, 3, 3), 
     (2, 3, 3),
     (1, 1, 2),
    (8, 1, 2),
     (2, 3, 3)
    ]

# 零件购买单价或成品装配成本
P = [(4, 18, 6)] * 6 # [(4, 18, 6), (4, 18, 6), (4, 18, 6), (4, 18, 6), (4, 18, 6), (4, 18, 6)]

# 市场售价
S = [56] * 6 # [56, 56, 56, 56, 56, 56]

# 不合格成品调换损失
R = [6, 6, 30, 30, 10, 10]

# 拆解费用
D = [5] * 5 + [40] # [5, 5, 5, 5, 5, 40]

# Mu: 零件1的购买良品率
# Lambda: 零件2的购买良品率
# Omega: 两良好零件下合成成功率
Mu = [1-0.1, 1-0.2, 1-0.1, 1-0.2, 1-0.1, 1-0.05] # [0.9, 0.8, 0.9, 0.8, 0.9, 0.95]
Lambda = [1-0.1, 1-0.2, 1-0.1, 1-0.2, 1-0.2, 1-0.05] # [0.9, 0.8, 0.9, 0.8, 0.8, 0.95]
Omega = [1-0.1, 1-0.2, 1-0.1, 1-0.2, 1-0.1, 1-0.05] # [0.9, 0.8, 0.9, 0.8, 0.9, 0.95]

# 为每个元组添加情况编号
situations = ['情况1', '情况2', '情况3', '情况4', '情况5', '情况6']
data_with_situations = [(situations[i],I[i], P[i], S[i], R[i], D[i], Mu[i], Lambda[i], Omega[i]) + tuple(decisions[i]) for i in range(6)]

# 创建 DataFrame
df = pd.DataFrame(data_with_situations, \
                  columns=['情况','检测成本','零件购买单价或成品装配成本', \
                           '市场售价', '不合格成品调换损失',\
                           '拆解费用', '零件1的购买良品率',\
                           '零件2的购买良品率','两良好零件下合成成功率',\
                            '是否检测零件1', '是否检测零件2', '是否检测成品', '是否拆解成品'])

# 将 DataFrame 保存为 Excel 文件
# df
df.to_excel('Result4_2.xlsx', index=False, sheet_name='最大化每百件商品利润的生产策略', engine='openpyxl')

print(df)