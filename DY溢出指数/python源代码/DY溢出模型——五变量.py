import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
from arch.unitroot import PhillipsPerron
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from statsmodels.tsa.vector_ar import var_model

# 忽略 InterpolationWarning
warnings.filterwarnings('ignore', category=InterpolationWarning)

# 读取 Excel 文件
excel_file = pd.ExcelFile("C:/Users/fufuQAQ/Desktop/课程论文/DY溢出指数模型/数据总表(2).xlsx")
df = excel_file.parse('BEKK')  # 将 Excel 文件中的指定工作表解析为一个 DataFrame 对象。类似于SQL表

# 保存时间序列列，假设时间序列列名为 '时间'，请根据实际情况修改
time_series = df['时间']

# 测试
# print(df)

# 对 BDI、BDTI、BCTI、WTI、GPR列进行对数化处理
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
    df[col] = df[col].ffill()  # 前向填充 NaN


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

# 输出 ADF 检验结果  ——————————————L2.1_修订，VAR稳定性检验——————————————
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

# ——————————————L2.1.2_修订 进行格林兰因果检验——————————————
# 1. 定义变量列表（与columns_to_log一致，确保顺序）
variables = columns_to_log  # ['BDI', 'BDTI', 'BCTI', 'WTI', 'GPR']

# 2. 初始化结果存储列表（记录：原假设、滞后阶数、p值、是否拒绝原假设）
granger_results = []

# 3. 遍历所有变量对，进行双向格兰杰检验
for target in variables:  # target：被解释变量（如"BDTI"）
    for cause in variables:  # cause：解释变量（如"BDI"）
        if target != cause:  # 排除"变量自身对自身的检验"（无意义）
            # 准备检验数据：[被解释变量, 解释变量]（grangercausalitytests要求的顺序）
            test_data = log_returns[[target, cause]].dropna()  # 使用对数收益率数据，并确保无缺失值

            # 执行格兰杰检验：maxlag=best_lags（仅检验最优滞后阶数，避免冗余）
            # verbose=False：不输出详细中间结果，仅通过返回值提取关键信息
            result = grangercausalitytests(
                x=test_data,
                maxlag=best_lags,
                addconst=True,  # 回归中加入常数项（默认True，符合VAR模型设定）
                # verbose=False  新版本已弃用该参数
            )

            # 提取该滞后阶数的检验结果（result的key为滞后阶数，取best_lags对应的结果）
            lag_result = result[best_lags]

            # 提取关键统计量：F检验的p值（常用且直观，也可选择卡方检验p值）
            # lag_result[0]是字典，包含'lrtest'（卡方）、'params_ftest'（F检验）等
            f_statistic = lag_result[0]['params_ftest'][0]
            p_value = lag_result[0]['params_ftest'][1]

            # 判断是否拒绝原假设（原假设："cause不是target的格兰杰原因"）
            alpha = 0.05  # 显著性水平（常用0.05）
            reject_null = p_value < alpha
            conclusion = "拒绝原假设（存在格兰杰因果关系）" if reject_null else "接受原假设（无格兰杰因果关系）"

            # 将结果存入列表
            granger_results.append({
                "被解释变量(Target)": target,
                "解释变量(Cause)": cause,
                "滞后阶数(Lag)": best_lags,
                "F统计量": round(f_statistic, 4),
                "p值": round(p_value, 4),
                "结论": conclusion
            })

# 4. 整理结果为DataFrame并打印
granger_df = pd.DataFrame(granger_results)
print("\n所有变量对的格兰杰因果检验结果：")
print(granger_df.to_string(index=False))  # 不显示行索引，更清晰

# 5. （可选）筛选显著的因果关系（p<0.05）
significant_granger = granger_df[granger_df["p值"] < 0.05]
print(f"\n显著的格兰杰因果关系（p<0.05）共{len(significant_granger)}组：")
if len(significant_granger) > 0:
    print(significant_granger.to_string(index=False))
else:
    print("无显著的格兰杰因果关系（所有p值≥0.05）")
# ——————————————L2.1.2_修订 进行格林兰因果检验——————————————

# ——————————————L2.1.3_修订 验证模型的稳定性——————————————
# model_test = VAR(log_returns)
# fitted_model = model_test.fit(maxlags=best_lags)
stability = results.is_stable()
print('模型是否稳定:')
print(stability)
# if stability:
#     print("模型稳定")
# else:
#     print("模型不稳定")
# ——————————————L2.1.3_修订 验证模型的稳定性——————————————


# ——————————————L2.1.4_修订 进行AR根检验——————————————
# 获取AR根并计算模
ar_roots = results.roots
roots_modulus = np.abs(ar_roots)
print("所有AR根的模：", np.round(roots_modulus, 4))  # 看全部根，而非前5个
print("最大AR根的模：", np.max(roots_modulus))  # 核心指标，需<1
is_stable_manual  = np.all(roots_modulus < 1)  # 所有根的模<1则稳定
print("手动判断稳定性（所有根模<1）：", is_stable_manual)
print("is_stable() 函数判断稳定性：", results.is_stable())

# 绘制AR根分布图
plt.figure(figsize=(10, 8))
# 绘制单位圆（判断稳定性的基准）
unit_circle = Circle((0, 0), 1, fill=False, color='black', linestyle='--', linewidth=1.5)
plt.gca().add_patch(unit_circle)
# 绘制AR根
plt.scatter(ar_roots.real, ar_roots.imag, color='#DC143C', s=60, alpha=0.8, label='AR-roots')
# 图形化
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
plt.xlim(-2.5, 2.5)  # 扩大x轴范围，便于观察不稳定的根
plt.ylim(-2.5, 2.5)
plt.xlabel('Real Part', fontsize=12) # 实部
plt.ylabel('Imaginary Part', fontsize=12) # 虚部
plt.title(f'VAR model AR-roots   p={best_lags}', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.axis('equal')  # 等比例坐标轴，确保单位圆为正圆
plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()

# 输出稳定性结果
print(f"模型稳定性：{'稳定' if is_stable_manual else '不稳定'}")
print(f"最大AR根的模：{np.max(roots_modulus):.4f}（需<1才稳定）")
print(f"所有AR根的模（前5个）：{np.round(roots_modulus[:5], 4)}")  # 显示前5个根的模
print("="*50)
# ——————————————L2.1.4_修订 进行AR根检验——————————————

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