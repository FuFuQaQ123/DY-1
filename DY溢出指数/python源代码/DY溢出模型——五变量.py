import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
from arch.unitroot import PhillipsPerron
import seaborn as sns

# 忽略 InterpolationWarning
warnings.filterwarnings('ignore', category=InterpolationWarning)

# 读取 Excel 文件
excel_file = pd.ExcelFile("C:/Users/fufuQAQ/Desktop/课程论文/DY溢出指数模型/数据总表(2).xlsx")
df = excel_file.parse('BEKK')  # 将 Excel 文件中的指定工作表解析为一个 DataFrame 对象。类似于SQL表

# 保存时间序列列，假设时间序列列名为 '时间'，请根据实际情况修改
time_series = df['时间']

# 测试
# print(df)

# 对 BDI、BDTI、BCTI、WTI 列进行对数化处理
columns_to_log = ['BDI', 'BDTI', 'BCTI', 'WTI', 'GPR']


# # 进行描述性统计
# descriptive_stats = df[columns_to_log].describe()
# print("描述性统计结果：")
# print(descriptive_stats)
#
# # 绘制折线图
# plt.figure(figsize=(12, 8))
# for col in columns_to_log:
#     plt.plot(time_series, df[col], label=col)
#
# # 总体
# plt.xlabel('Time')
# plt.xticks(rotation=45)
# plt.ylabel('values')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
#
# # 分开绘制每一列的折线图
# for col in columns_to_log:
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_series, df[col])
#     plt.title(f'{col}')
#     plt.xlabel('Time')
#     plt.xticks(rotation=45)
#     plt.ylabel('values')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

for col in columns_to_log:
    df[col] = np.where(df[col] > 0, np.log(df[col]), np.nan)  # 对数化处理
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)  # 替换 inf 为 NaN
    df[col] = df[col].fillna(method='ffill')  # 前向填充 NaN


df = df.dropna()
# 测试
# print(df)

# 第二步：计算对数收益率
log_returns = (df[columns_to_log].diff() * 100).dropna() # 数据放大否  * 100
# df = df.dropna()  # 删除对数化处理后可能产生的 NaN
# df[columns_to_log] = log_returns  # 将对数收益率覆盖原始列  将对数收益率添加回原始 DataFrame


# 测试
# print(log_returns)



# ADF 检验函数
def adf_test(column_data, column_name):
    if np.all(column_data == column_data.iloc[0]):
        print(f"Column {column_name} is a constant sequence. Skipping ADF test.")
        return None

    """
    result 是一个包含检验结果的元组，通常包含以下内容：
    检验统计量（ADF Statistic）
    p 值
    使用的滞后阶数
    用于检验的样本数量
    不同置信水平下的临界值（例如 1%，5%，10%）
    """
    result = adfuller(column_data)
    adf_statistic = result[0]
    critical_value_1_percent = result[4]['1%']
    return [column_name, adf_statistic, critical_value_1_percent]


# PP 检验函数
def pp_test(column_data, column_name):
    result = PhillipsPerron(column_data)
    pp_statistic = result.stat
    critical_value_1_percent = result.critical_values['1%']
    return [column_name, pp_statistic, critical_value_1_percent] # 时间序列数据的名称。PP 检验统计量。1% 置信水平下的临界值。


# 存储 ADF 和 PP 的列表
adf_results = []
pp_results = []

# 对各列进行 ADF 和 PP 检验并存储结果
for col in columns_to_log:
    adf_result = adf_test(log_returns[col], col)
    if adf_result:
        adf_results.append(adf_result)

    pp_result = pp_test(log_returns[col], col)
    pp_results.append(pp_result)

# 输出 ADF 检验结果
print("ADF 检验结果:")
adf_df = pd.DataFrame(adf_results, columns=['列名', 'ADF 值', '1% 显著性水平临界值'])
print(adf_df)

# 输出 PP 检验结果
print("\nPP 检验结果:")
pp_df = pd.DataFrame(pp_results, columns=['列名', 'PP 值', '1% 显著性水平临界值'])
print(pp_df)


        # epsilon = 1e-10
        # # 2. 数据预处理：计算对数收益率
        # log_returns = (df[columns_to_log].clip(lower=epsilon) - df[columns_to_log].shift(1).clip(lower=epsilon)).dropna()

# 同时对时间序列列进行相同的操作，保证索引一致
time_series = time_series[log_returns.index]

# 将时间序列列设置为 log_returns 的索引
log_returns.index = time_series[log_returns.index]

        # 3. 平稳性检验（ADF 检验）（表 1）
        # print("\n对数收益率的 ADF 检验结果:")
        # for col in log_returns.columns:
        #     if not np.all(log_returns[col] == log_returns[col].iloc[0]):
        #         result = adfuller(log_returns[col])
        #         print(f"{col}: ADF Statistic = {result[0]}, p-value = {result[1]}")

# 确保索引单调
log_returns = log_returns.sort_index()

# print(log_returns)

# 确保索引是 DatetimeIndex 或 PeriodIndex 并设置频率
if not isinstance(log_returns.index, pd.DatetimeIndex):
    log_returns.index = pd.to_datetime(log_returns.index)

# 使用 to_period() 方法设置频率
log_returns.index = log_returns.index.to_period('M')

# print(log_returns)

# 4. 构建 VAR 模型
model = VAR(log_returns)
selected_lags = model.select_order(maxlags=10)
# print(selected_lags.summary())
"""
这是一个方法，返回一个摘要报告，显示不同滞后期数下的信息准则值。
输出结果通常包括：
AIC（赤池信息准则）：越小越好。
BIC（贝叶斯信息准则）：越小越好。
FPE（最终预测误差）：越小越好。
HQIC（汉南-昆信息准则）：越小越好。
通常选择 AIC 或 BIC 最小的滞后期数作为最佳滞后期数。
"""
best_lags = selected_lags.aic # 提取基于 AIC 准则选择的最佳滞后期数。
results = model.fit(maxlags=best_lags)

print(f"滞后阶数p = {best_lags}") # L1.1_修订  VAR滞后阶数p，通过AIC准则自动计算最佳滞后阶数，P = 6

# print(results.summary())
"""
拟合完成后，results 对象提供了多种方法和属性，用于分析模型结果。以下是一些常用的方法：
(1) 查看模型摘要:          print(results.summary())
(2) 脉冲响应分析（IRF）:   irf = results.irf(5)  # 计算 5 期的脉冲响应
                        irf.plot()
(3) 方差分解（FEVD）      fevd = results.fevd(5)  # 计算 5 期的方差分解,分析每个变量对系统方差的贡献
                        fevd.plot() # 会生成方差分解的图表。
(4) 预测                  forecast = results.forecast(log_returns.values[-best_lags:], steps=5)
"""

# print(best_lags)  # 20
# print(results.summary())



# 动态计算
# 滚动窗口大小
rolling_window = 45  # L1.3_修订  时变溢出的滚动窗口长度为45

# 初始化存储溢出指数的列表
bdi_to_bdi = []
bdi_to_bdti = []
bdi_to_bcti = []
bdi_to_wti = []
bdi_to_gpr = []

bdti_to_bdi = []
bdti_to_bdti = []
bdti_to_bcti = []
bdti_to_wti = []
bdti_to_gpr = []

bcti_to_bdi = []
bcti_to_bdti = []
bcti_to_bcti = []
bcti_to_wti = []
bcti_to_gpr = []

wti_to_bdi = []
wti_to_bdti = []
wti_to_bcti = []
wti_to_wti = []
wti_to_gpr = []

gpr_to_bdi = []
gpr_to_bdti = []
gpr_to_bcti = []
gpr_to_wti = []
gpr_to_gpr = []

# 初始化存储总溢出指数的列表
total_spillover = []

window_start_times = []  # 用于存储每个滚动窗口的起始时间

# 对对数收益率数据 log_returns 进行滚动窗口分析
for i in range(len(log_returns) - rolling_window):
    # 提取滚动窗口数据

    window_data = log_returns.iloc[i:i + rolling_window]

    # 拟合 VAR 模型
    model = VAR(window_data)

    window_model = model.fit(maxlags=selected_lags.aic)

    # L2.1_修订  VAR稳定性（特征值模数）
    # 先进行稳定性检验（放在FEVD计算前，如果稳定，则进入FEVD计算）
    # 计算当前窗口模型的特征值模数
    companion = window_model.companion_matrix
    eig_mod = np.abs(np.linalg.eigvals(companion))
    is_stable = np.all(eig_mod < 1)

    print(f"窗口{i}稳定性：{'稳定' if is_stable else '不稳定'}")
    # L2.1_修订  VAR稳定性（特征值模数）


    # 计算 FEVD
    fevd = window_model.fevd(5) # L1.2_修订  FEVD的预测水平H，手动设定为5

    # 引入GPR后矩阵非正定？
    try:
        # 提取最后一个预测步长的方差分解矩阵（第二维）
        fevd_decomp = fevd.decomp[:, -1, :]  # 形状：(4, 4)

        # 获取变量索引
        bdi_index = list(log_returns.columns).index('BDI')
        bdti_index = list(log_returns.columns).index('BDTI')
        bcti_index = list(log_returns.columns).index('BCTI')
        wti_index = list(log_returns.columns).index('WTI')
        gpr_index = list(log_returns.columns).index('GPR')

        # 提取溢出指数
        bdi_to_bdi.append(fevd_decomp[bdi_index][bdi_index])
        bdi_to_bdti.append(fevd_decomp[bdi_index][bdti_index])
        bdi_to_bcti.append(fevd_decomp[bdi_index][bcti_index])
        bdi_to_wti.append(fevd_decomp[bdi_index][wti_index])
        bdi_to_gpr.append(fevd_decomp[bdi_index][gpr_index])

        bdti_to_bdi.append(fevd_decomp[bdti_index][bdi_index])
        bdti_to_bdti.append(fevd_decomp[bdti_index][bdti_index])
        bdti_to_bcti.append(fevd_decomp[bdti_index][bcti_index])
        bdti_to_wti.append(fevd_decomp[bdti_index][wti_index])
        bdti_to_gpr.append(fevd_decomp[bdti_index][gpr_index])

        bcti_to_bdi.append(fevd_decomp[bcti_index][bdi_index])
        bcti_to_bdti.append(fevd_decomp[bcti_index][bdti_index])
        bcti_to_bcti.append(fevd_decomp[bcti_index][bcti_index])
        bcti_to_wti.append(fevd_decomp[bcti_index][wti_index])
        bcti_to_gpr.append(fevd_decomp[bcti_index][gpr_index])

        wti_to_bdi.append(fevd_decomp[wti_index][bdi_index])
        wti_to_bdti.append(fevd_decomp[wti_index][bdti_index])
        wti_to_bcti.append(fevd_decomp[wti_index][bcti_index])
        wti_to_wti.append(fevd_decomp[wti_index][wti_index])
        wti_to_gpr.append(fevd_decomp[wti_index][gpr_index])

        gpr_to_bdi.append(fevd_decomp[gpr_index][bdi_index])
        gpr_to_bdti.append(fevd_decomp[gpr_index][bdti_index])
        gpr_to_bcti.append(fevd_decomp[gpr_index][bcti_index])
        gpr_to_wti.append(fevd_decomp[gpr_index][wti_index])
        gpr_to_gpr.append(fevd_decomp[gpr_index][gpr_index])


        # 计算总溢出指数
        total_spillover.append((np.sum(fevd_decomp) - np.trace(fevd_decomp)) / 5)

        window_start_times.append(log_returns.index[i].to_timestamp())

    except AttributeError:
        print(f"处理窗口 {i} 时，fevd 对象没有 'decomp' 属性。")

# 计算平均方向性溢出指数
average_spillover_matrix = {
    'BDI_to_BDI': np.mean(bdi_to_bdi),
    'BDI_to_BDTI': np.mean(bdi_to_bdti),
    'BDI_to_BCTI': np.mean(bdi_to_bcti),
    'BDI_to_WTI': np.mean(bdi_to_wti),
    'BDI_to_GPR': np.mean(bdi_to_gpr),

    'BDTI_to_BDI': np.mean(bdti_to_bdi),
    'BDTI_to_BDTI': np.mean(bdti_to_bdti),
    'BDTI_to_BCTI': np.mean(bdti_to_bcti),
    'BDTI_to_WTI': np.mean(bdti_to_wti),
    'BDTI_to_GPR': np.mean(bdti_to_gpr),

    'BCTI_to_BDI': np.mean(bcti_to_bdi),
    'BCTI_to_BDTI': np.mean(bcti_to_bdti),
    'BCTI_to_BCTI': np.mean(bcti_to_bcti),
    'BCTI_to_WTI': np.mean(bcti_to_wti),
    'BCTI_to_GPR': np.mean(bcti_to_gpr),

    'WTI_to_BDI': np.mean(wti_to_bdi),
    'WTI_to_BDTI': np.mean(wti_to_bdti),
    'WTI_to_BCTI': np.mean(wti_to_bcti),
    'WTI_to_WTI': np.mean(wti_to_wti),
    'WTI_to_GPR': np.mean(wti_to_gpr),

    'GPR_to_BDI': np.mean(gpr_to_bdi),
    'GPR_to_BDTI': np.mean(gpr_to_bdti),
    'GPR_to_BCTI': np.mean(gpr_to_bcti),
    'GPR_to_WTI': np.mean(gpr_to_wti),
    'GPR_to_GPR': np.mean(gpr_to_gpr)
}

# 计算平均总溢出指数
average_total_spillover = np.mean(total_spillover)

# 输出平均方向性溢出指数
print("\n平均方向性溢出指数 (Average Spillover Index):")
for key, value in average_spillover_matrix.items():
    print(f"{key}: {value:.4f}")

# 输出平均总溢出指数
print("\n平均总溢出指数 (Average Total Spillover Index):")
print(f"Average Total Spillover Index: {average_total_spillover:.4f}")
# len all = 6239



# ————————————————TO————————————————
AVG_BDI_To = []
AVG_BDTI_To = []
AVG_BCTI_To = []
AVG_WTI_To = []
AVG_GPR_To = []

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_bdti[i] + bdi_to_bcti[i] + bdi_to_wti[i] + bdi_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BDI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdti_to_bdi[i] + bdti_to_bcti[i] + bdti_to_wti[i] + bdti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BDTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bcti_to_bdi[i] + bcti_to_bdti[i] + bcti_to_wti[i] + bcti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BCTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = wti_to_bdi[i] + wti_to_bdti[i] + wti_to_bcti[i] + wti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_WTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = gpr_to_bdi[i] + gpr_to_bdti[i] + gpr_to_bcti[i] + gpr_to_wti[i]
    avg_value = sum_value / 4
    AVG_GPR_To.append(avg_value)

# ————————————————TO————————————————



# ————————————————FROM————————————————
AVG_BDI_FROM = []
AVG_BDTI_FROM = []
AVG_BCTI_FROM = []
AVG_WTI_FROM = []
AVG_GPR_FROM = []

for i in range(len(bdi_to_bdi)):
    sum_value =  bdti_to_bdi[i] + bcti_to_bdi[i] + wti_to_bdi[i] + gpr_to_bdi[i]
    avg_value = sum_value / 4
    AVG_BDI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_bdti[i] + bcti_to_bdti[i] + wti_to_bdti[i] + gpr_to_bdti[i]
    avg_value = sum_value / 4
    AVG_BDTI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value =  bdi_to_bcti[i] + bdti_to_bcti[i] +  wti_to_bcti[i] + gpr_to_bcti[i]
    avg_value = sum_value / 4
    AVG_BCTI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_wti[i] + bdti_to_wti[i] + bcti_to_wti[i] + gpr_to_wti[i]
    avg_value = sum_value / 4
    AVG_WTI_FROM.append(avg_value)


for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_gpr[i] + bdti_to_gpr[i] + bcti_to_gpr[i] + wti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_GPR_FROM.append(avg_value)
# ————————————————FROM————————————————

# 总溢出指数图
plt.figure(figsize=(12, 8))  # 显示大小
# plt.subplot(2, 2, 1)
plt.plot(window_start_times, total_spillover, label=' total_spillover ', color='blue')
plt.title(' total_spillover')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()


# 绘制 BDI-TO 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BDI_To, label='BDI to', color='blue', alpha=0.3)
plt.title('BDI_To')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 BDTI-TO 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BDTI_To, label='BDTI to', color='blue', alpha=0.3)
plt.title('BDTI_To')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 BCTI-TO 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BCTI_To, label='BCTI to', color='blue', alpha=0.3)
plt.title('BCTI_To')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 WTI-TO 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_WTI_To, label='WTI to', color='blue', alpha=0.3)
plt.title('WTI_To')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 GPR-TO 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_GPR_To, label='GPR_To', color='blue', alpha=0.3)
plt.title('GPR_To')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()


# ——————————————————split————————————————————

# 绘制 BDI-FROM 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BDI_FROM, label='BDI from', color='blue', alpha=0.3)
plt.title('BDI_FROM')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 BDTI-FROM 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BDTI_FROM, label='BDTI from', color='blue', alpha=0.3)
plt.title('BDTI_FROM')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 BCTI-FROM 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_BCTI_FROM, label='BCTI from', color='blue', alpha=0.3)
plt.title('BCTI_FROM')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()

# 绘制 WTI-FROM 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_WTI_FROM, label='WTI from', color='blue', alpha=0.3)
plt.title('WTI_FROM')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()


# 绘制 WTI-FROM 图
plt.figure(figsize=(12, 8))  # 显示大小
plt.fill_between(window_start_times, AVG_GPR_FROM, label='GPR_FROM', color='blue', alpha=0.3)
plt.title('GPR_FROM')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()


plt.tight_layout()  # 自动调整布局
plt.show()