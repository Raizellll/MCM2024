'''问题二：一道工序和两个零件下的生产过程决策'''

import pandas as pd
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

    return X4 * (N / w) * (1 - w) * D

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
        recycle = X4 * (N / w) * (1 - w)
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

def profit_inner(N, X1, X2, X3, X4, I1, I2, I3, P1, P2, P3, S, R, D, w, Mu, Lambda, a, b):
    return sale(N, S) - check_last(X3, N, w, I3) - replace_last(X3, N, w, R) - \
    check_1(X1, N, w, I1, Mu) - check_2(X2, N, w, I2, Lambda) - disassemble(X4, N, w, D) - \
    purchase_1(a, P1) - purchase_2(b, P2) - assemble(P3, N, w)

def profit(N, X, I, P, S, R, D, Mu, Lambda, Omega):
    '''
    return 总利润(total_profit)
    
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

# 穷举求出某个情况的最优解

def get_best(I, P, S, R, D, Mu, Lambda, Omega):
    max_profit = float('-inf')
    best_case = None
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    one_profit = profit(100, (i, j, k, l), I, P, S, R, D, Mu, Lambda, Omega)
                    if one_profit > max_profit:
                        best_case = (i, j, k, l)
                        max_profit = one_profit
                    
    return max_profit, best_case

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

best_cases = []
for i in range(6):
    max_profit, best_case = get_best(I[i], P[i], S[i], R[i], D[i], Mu[i], Lambda[i], Omega[i])
    best_cases.append(best_case)

best_cases_with_profit = []
for i in range(6):
    max_profit, best_case = get_best(I[i], P[i], S[i], R[i], D[i], Mu[i], Lambda[i], Omega[i])
    best_cases_with_profit.append((max_profit,) + best_case)

# 为每个元组添加情况编号
situations = ['情况1', '情况2', '情况3', '情况4', '情况5', '情况6']
data_with_situations = [(situations[i],I[i], P[i], S[i], R[i], D[i], Mu[i], Lambda[i], Omega[i]) + best_cases_with_profit[i] for i in range(6)]

# 创建 DataFrame
df = pd.DataFrame(data_with_situations, \
                  columns=['情况','检测成本','零件购买单价或成品装配成本', \
                           '市场售价', '不合格成品调换损失',\
                           '拆解费用', '零件1的购买良品率',\
                           '零件2的购买良品率','两良好零件下合成成功率',\
                           '总收益', '是否检测零件1', '是否检测零件2', '是否检测成品', '是否拆解成品'])

print(df)

# 将 DataFrame 保存为 Excel 文件
df.to_excel('Result2.xlsx', index=False, sheet_name='最大化每百件商品利润的生产策略', engine='openpyxl')
