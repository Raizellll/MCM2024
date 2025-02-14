'''问题三：复杂工序和多个零件下的生产过程决策'''

import pandas as pd

def profit_inner(N, S, R, D_final, P_final, I_final, u_final, x_final, y_final, m_semi, D_semi, P_semi, I_semi, u_semi, x_semi, y_semi, m_part, P_part, I_part, l_part, x_part):
    income = N * S # 营收income
    final_assemble = N / u_final * P_final # 成品装配final_assemble
    final_test = N / u_final * I_final * x_final # 成品检测final_test
    final_recycle = (N / u_final) * ((1 - u_final) / u_final) * R * (1 - x_final) # 成品回收final_recycle
    final_disassemble = N * (1 / u_final - 1) * D_final * y_final # 成品拆解final_disassemble
    
    semi_assemble = 0 # '半成品装配semi_assemble
    for j in range(3):
        semi_assemble += m_semi[j] * P_semi[j]
    
    semi_test = 0 # 半成品检测semi_test
    for j in range(3):
        semi_test += N / u_final / u_semi[j] * I_semi[j] * x_semi[j]
    
    semi_disassemble = 0 # 半成品拆解semi_disassemble
    for j in range(3):
        semi_disassemble += N / u_final / u_semi[j] * (1 - u_semi[j]) * x_semi[j] * y_semi[j] * D_semi[j]
    
    part_purchase = 0 # 零件购买part_purchase
    for i in range(8):
        part_purchase += m_part[i] * P_part[i]
    
    part_test = 0 # 零件检测part_test
    for i in range(8):
        j = i // 3
        part_test += m_semi[j] / l_part[i] * I_part[i] * x_part[i]
    
    return income - final_assemble - final_test - final_recycle - final_disassemble - \
    semi_assemble - semi_test - semi_disassemble - part_purchase - part_test

# part: i, 1-8 (0-7)
# semi: j, 1-3 (0-2)
# 零件

# x_part: 零件是否检测
# l_part: 零件购买良品率
# u_part: 零件最终良品率

def get_u_part(x_part, l_part):
    u_part = [] # [1, 0.9, 1, 0.9, 1, 1, 1, 0.9]
    for i in range(8):
        if x_part[i] == 0:
            u_part.append(l_part[i])
        else:
            u_part.append(1)
    return u_part

def get_m_part(N, u_final, u_semi, m_semi, l_part, x_part, x_semi, y_semi):
    # 购入数量 = 需求 - 回收
    def get_n_part():  # 需求
        n_part = []
        for i in range(8):
            j = min(i // 3, 2)  # 确保 j 的最大值为 2
            if x_part[i] == 0:
                n_part.append(m_semi[j])
            else:
                n_part.append(m_semi[j] / l_part[i])
        return n_part

    def get_r_part():  # 回收
        r_part = []
        for i in range(8):
            j = min(i // 3, 2)  # 确保 j 的最大值为 2
            r_part.append(N * x_semi[j] * y_semi[j] * (1 - u_semi[j]) / u_final / u_semi[j])
        return r_part

    n_part = get_n_part()
    r_part = get_r_part()
    m_part = [(n_part[i] - r_part[i]) for i in range(8)]
    return m_part


# 半成品
def get_u_semi(l_semi, u_part):
    u_semi = l_semi[:]
    for i in range(8):
        j = i // 3
        u_semi[j] *= u_part[i]
    return u_semi # [0.9, 0.81, 0.81]

def get_w_semi(x_semi, u_semi):
    w_semi = []
    for j in range(3):
        if x_semi[j] == 0:
            val = u_semi[j]
        else:
            val = 1
        w_semi.append(val)
    return w_semi

def get_m_semi(N, u_final, y_final, x_semi, u_semi):
    # 购入数量 = 需求 - 回收
    def get_n_semi(): # 需求
        n_semi = []
        for j in range(3):
            if x_semi[j] == 0:
                n_semi.append(N / u_final)
            else:
                n_semi.append(N / u_final / u_semi[j])
        return n_semi

    def get_r_semi(): # 回收
        r_semi = []
        for j in range(3):
            r_semi.append((N / u_final - N) * y_final)
        return r_semi

    n_semi = get_n_semi()
    r_semi = get_r_semi()
    m_semi = [(n_semi[j] - r_semi[j]) for j in range(3)]
    return m_semi

# 成品
def get_u_final(w_semi, l_final):
    u_final = 1
    for j in range(3):
        u_final *= w_semi[j]
    u_final *= l_final
    return u_final

# 已知条件假设eg.
# 是否检测
x_part = [1, 1, 1, 0, 1, 1, 1, 0]
x_semi = [1, 0, 1]
x_final = 0

# 是否拆解
y_semi = [1, 1, 1]
y_final = 1

# 良品率
l_part = [1 - 0.1] * 8 # [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
l_semi = [1 - 0.1] * 3 # [0.9, 0.9, 0.9] # 半成品合成良品率
l_final = 1 - 0.1

N = 100 # 假设的数量
S = 200 # 市场售价
R = 40 # 调换损失

# 成品
D_final = 10 # 拆解费用
P_final = 8 # 装配成本
I_final = 6 # 检测成本

# 半成品
D_semi = [6] * 3 # 拆解费用
P_semi = [8] * 3 # 装配成本
I_semi = [4] * 3 # 检测成本

# 零配件
P_part = [2, 8, 12, 2, 8, 12, 8, 12] # 购买单价
I_part = [1, 1, 2, 1, 1, 2, 1, 2] # 检测成本

u_part = get_u_part(x_part, l_part)
u_semi = get_u_semi(l_semi, u_part)
w_semi = get_w_semi(x_semi, u_semi)
u_final = get_u_final(w_semi, l_final)
m_semi = get_m_semi(N, u_final, y_final, x_semi, u_semi)
m_part = get_m_part(N, u_final, u_semi, m_semi, l_part, x_part, x_semi, y_semi)


def profit(N, S, R, l, P, D, I, x, y):
    # print('数量N', N)
    # print('售价S', S)
    # print('调换成本R', R)
    # print('良品率l', l)
    # print('成本/组装费P', P)
    # print('拆分成本D', D)
    # print('检测费I', I)
    # print('是否检测x', x)
    # print('是否拆分y', y)

    I_part, I_semi, I_final = I # 检测费
    P_part, P_semi, P_final = P # 成本/组装费
    l_part, l_semi, l_final = l # 零件购买良品率,半成品合成良品率,成品合成良品率
    x_part, x_semi, x_final = x # 是否检测
    D_semi, D_final = D # 拆分成本
    y_semi, y_final = y # 是否拆分
    
    u_part = get_u_part(x_part, l_part) # 零件最终良品率(经检测)
    u_semi = get_u_semi(l_semi, u_part) # 半成品最终良品率(未经检测)
    w_semi = get_w_semi(x_semi, u_semi) # 半成品检测后良品率(基于半成品最终良品率)
    u_final = get_u_final(w_semi, l_final) # 成品最终良品率

    # 倒推数量
    m_semi = get_m_semi(N, u_final, y_final, x_semi, u_semi) # 半成品购入数量
    m_part = get_m_part(N, u_final, u_semi, m_semi, l_part, x_part, x_semi, y_semi) # 零件购入数量

    profit = profit_inner(N, S, R, D_final, P_final, I_final, u_final, x_final, y_final, m_semi, D_semi, \
                        P_semi, I_semi, u_semi, x_semi, y_semi, m_part, P_part, I_part, l_part, x_part)
    return profit

def profit_case(N, x, y):
    '''代入题设条件的profit函数,用于模型中,x,y作为可变参数通过模型确定,N是目标数量'''
    x_part, x_semi, x_final = x # 零配件,半成品,成品是否检测
    # x_part是包含8个元素的列表,x_semi是包含3个元素的列表,x_final是1个元素
    y_semi, y_final = y # 半成品,成品是否拆分
    # y_semi是包含3个元素的列表,y_final是1个元素
    
    return profit(N, 200, 40, ([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9], 0.9), \
       ([2, 8, 12, 2, 8, 12, 8, 12], [8, 8, 8], 8), ([6, 6, 6], 10), \
      ([1, 1, 2, 1, 1, 2, 1, 2], [4, 4, 4], 6), x, y)

import itertools

def find_best_strategy(N):
    best_profit = float('-inf')  # 用于记录最大利润
    best_strategy = None  # 用于记录最佳策略

    # 所有可能的检测策略组合 (零配件8个, 半成品3个, 成品1个)
    detection_combinations = list(itertools.product([0, 1], repeat=12))  # 8 + 3 + 1

    # 所有可能的拆解策略组合 (半成品3个, 成品1个)
    disassemble_combinations = list(itertools.product([0, 1], repeat=4))  # 3 + 1

    for detection in detection_combinations:
        for disassemble in disassemble_combinations:
            x_part = detection[:8]  # 前8个是零配件检测策略
            x_semi = detection[8:11]  # 中间3个是半成品检测策略
            x_final = detection[11]  # 最后1个是成品检测策略
            y_semi = disassemble[:3]  # 前3个是半成品拆解策略
            y_final = disassemble[3]  # 最后1个是成品拆解策略

            # 计算当前组合下的利润
            current_profit = profit_case(N, (x_part, x_semi, x_final), (y_semi, y_final))

            # 更新最大利润和最优策略
            if current_profit > best_profit:
                best_profit = current_profit
                best_strategy = {
                    'x_part': x_part,
                    'x_semi': x_semi,
                    'x_final': x_final,
                    'y_semi': y_semi,
                    'y_final': y_final,
                    'profit': current_profit
                }

    return best_strategy

# 测试N = 100的最优策略
best_strategy = find_best_strategy(100)
print("最优策略:", best_strategy)


import itertools

def find_optimal_strategy(N):
    # 初始化最大利润
    max_profit = float('-inf')
    best_strategy = None
    
    # 穷举所有可能的组合
    for x_part in itertools.product([0, 1], repeat=8):
        for x_semi in itertools.product([0, 1], repeat=3):
            for x_final in [0, 1]:
                for y_semi in itertools.product([0, 1], repeat=3):
                    for y_final in [0, 1]:
                        # 当前组合的决策变量
                        x = (x_part, x_semi, x_final)
                        y = (y_semi, y_final)
                        
                        # 计算当前组合的利润
                        current_profit = profit_case(N, x, y)
                        
                        # 如果当前利润大于最大利润，更新最优策略
                        if current_profit > max_profit:
                            max_profit = current_profit
                            best_strategy = (x, y)
    
    # 返回最优策略及其对应的最大利润
    return best_strategy, max_profit

# 计算并输出最优策略
N = 100  # 假设生产100个成品
optimal_strategy, max_profit = find_optimal_strategy(N)
print("最优策略:", optimal_strategy)
print("最大利润:", max_profit)


# 扁平化数据
flattened_data = []
flattened_data.extend(x_part)
flattened_data.extend(x_semi)
flattened_data.append(x_final)
flattened_data.extend(y_semi)
flattened_data.append(y_final)
N = 100
flattened_data.append(N)
flattened_data.append(max_profit)

# 创建 DataFrame
df = pd.DataFrame([flattened_data], columns=[
    '是否检测零件1', '是否检测零件2', '是否检测零件3', '是否检测零件4',
    '是否检测零件5', '是否检测零件6', '是否检测零件7', '是否检测零件8',
    '是否检测半成品1', '是否检测半成品2', '是否检测半成品3',
    '是否检测成品', '是否拆分半成品1', '是否拆分半成品2', '是否拆分半成品3',
    '是否拆分成品', '售出正品数量','最大利润'
])

# 保存到 Excel 文件
df.to_excel('Result3.xlsx',sheet_name='最大化每百件商品利润的生产策略', index=False, header=True)