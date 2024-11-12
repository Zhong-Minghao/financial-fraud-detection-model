import numpy as np
import pandas as pd
from collections import Counter
import re
import os
import glob
import sys

# ===========================import the data============================
# 编码检测
# with open(csv_files[0], 'rb') as f:
#     encoding_type = chardet.detect(f.read())

# start_year = 2001
# end_year = 2022

start_year = int(sys.argv[1])
end_year = int(sys.argv[2])
folder_path = sys.argv[3]


print('==============正在导入违规表数据==================')
df = pd.read_excel('./input/STK_Violation_Main.xlsx', header=0)
df = df.iloc[2:]
print(df.head())
stock_num = pd.read_excel(folder_path + '/行业指数成分及价格信息/stock_nums全A.xlsx', sheet_name='中信一级', index_col=0, header=1)
vio_summary = pd.read_excel('./output/违规案例整理.xlsx', header=0, index_col=0)
print('==================导入完成======================')

vio_summary = vio_summary.loc[vio_summary['label'] == 1].copy()
vio_summary = vio_summary.reset_index(drop=True)

chinese_chars = pd.DataFrame({'所属行业': ['CI005029.WI', 'CI005026.WI', 'CI005028.WI', 'CI005020.WI', 'CI005022.WI', 'CI005007.WI', 'CI005017.WI', 'CI005008.WI', 'CI005002.WI', 'CI005027.WI', 'CI005009.WI', 'CI005019.WI', 'CI005025.WI', 'CI005013.WI', 'CI005004.WI', 'CI005023.WI', 'CI005010.WI', 'CI005005.WI', 'CI005021.WI', 'CI005030.WI', 'CI005001.WI', 'CI005011.WI', 'CI005015.WI', 'CI005003.WI', 'CI005018.WI', 'CI005014.WI', 'CI005012.WI', 'CI005006.WI', 'CI005016.WI', 'CI005024.WI'],
                                    'industries': ['综合', '通信', '传媒', '农林牧渔', '非银行金融', '建筑', '纺织服装', '建材', '煤炭', '计算机', '轻工制造', '食品饮料', '电子', '汽车', '电力及公用事业', '房地产', '机械', '钢铁', '银行', '综合金融', '石油石化', '电力设备及新能源', '消费者服务', '有色金属', '医药', '商贸零售', '国防军工', '基础化工', '家电', '交通运输']})
stock_num.columns = stock_num.columns.str[:-4]
vio_summary = vio_summary.merge(chinese_chars, left_on='所属行业', right_on='所属行业', how='left')
vio_summary.drop(columns=['所属行业'], inplace=True)
vio_summary = vio_summary.rename(columns={'industries': '所属行业'})

# ===========================================统计违规行业分布=================================================
vio_summary_pivot = vio_summary.pivot_table(index='年份', columns='所属行业', aggfunc='size', fill_value=0)

for this_industry in set(stock_num.columns).difference(vio_summary_pivot.columns):
    vio_summary_pivot[this_industry] = 0

stock_num_annual = stock_num.loc[stock_num.index % 100 == 12].copy()
stock_num_annual.index = stock_num_annual.index // 100
stock_num_annual = stock_num_annual.reindex(columns=vio_summary_pivot.columns)

industries_fraud_pct = vio_summary_pivot / stock_num_annual

industries_fraud_count = vio_summary_pivot.sum(axis=0).reset_index()
industries_fraud_count.columns = ['所属行业', 'Count']
industries_fraud_count = industries_fraud_count.sort_values(by='Count', ascending=False)

# ===========================================造假年份分布====================================================
violate_year_dist = vio_summary.groupby(by='年份').count()
violate_year_dist = violate_year_dist['证券代码'].copy()
violate_year_dist = violate_year_dist.to_frame(name='vio_num')
stock_num_allA = stock_num.sum(axis=1)
stock_num_allA = stock_num_allA.loc[stock_num_allA.index % 100 == 12]
stock_num_allA.index = stock_num_allA.index // 100
stock_num_allA = stock_num_allA.to_frame(name='stock_num')
vio_num_ratio = pd.merge(violate_year_dist, stock_num_allA, left_index=True, right_index=True, how='left')
vio_num_ratio = vio_num_ratio.loc[~vio_num_ratio['stock_num'].isna()].copy()
vio_num_ratio['violate_rate'] = vio_num_ratio['vio_num'] / vio_num_ratio['stock_num']
new_column_order = ['vio_num', 'violate_rate', 'stock_num']
vio_num_ratio = vio_num_ratio[new_column_order].copy()

# ===========================================违规持续时间====================================================
def get_duration_years(years):
    if np.isnan(years).all():
        return []
    duration = []
    begin_year = years[0]  # 初始化第一个年份
    for i in range(1, len(years)):
        if years[i] != years[i - 1] + 1:  # 如果当前年份不是前一个年份下一年
            duration.append(years[i-1]-begin_year+1)
            begin_year = years[i]
    duration.append(years[-1]-begin_year+1)

    return duration


duration_df = vio_summary.copy()
all_dur = []
company_list = duration_df['证券代码'].unique()
for company_name in company_list:
    tmp = duration_df.loc[duration_df['证券代码'] == company_name].copy()
    all_vio_year = tmp['年份'].dropna().values.tolist()
    all_years = sorted(set(all_vio_year))
    all_dur.append(get_duration_years(all_years))
all_dur_list = []
for i in all_dur:
    all_dur_list = all_dur_list + i

duration_each_stock_df = pd.DataFrame(company_list, columns=['证券代码'])
duration_each_stock_df['持续时间'] = all_dur
duration_each_stock_df = duration_each_stock_df.explode('持续时间')
duration_each_stock_df = duration_each_stock_df.sort_values(by='持续时间', ascending=False)
duration_each_stock_df_output = duration_each_stock_df.loc[duration_each_stock_df['持续时间'] >= 10].copy()

based_law = []
stock_name = []
for i, row in duration_each_stock_df_output.iterrows():
    tmp = df.loc[df['Symbol'] == row['证券代码'][0:-3]]
    this_law = tmp['DocumentNumber'].values.tolist()
    this_law = [x for x in this_law if not pd.isna(x)]
    this_law = this_law[0] if len(this_law) == 1 else '\n'.join(this_law)
    based_law.append(this_law)
    this_name = tmp['ShortName'].unique().tolist()
    this_name = this_name[0] if len(this_name) == 1 else '\n'.join(this_name)
    stock_name.append(this_name)
duration_each_stock_df_output['证券简称'] = stock_name
duration_each_stock_df_output['判定依据'] = based_law

counter = Counter(all_dur_list)
fraud_duration = pd.DataFrame(counter.items(), columns=['Element', 'Count'])
fraud_duration = fraud_duration.sort_values(by='Element', ascending=True)
# fraud_duration.to_csv('count_fraud_duration.csv')


# ============================================统计违规类型==============================================
violation_dict = {
    'P2501': '虚构利润',
    'P2502': '虚列资产',
    'P2503': '虚假记载(误导性陈述)',
    'P2504': '推迟披露',
    'P2505': '重大遗漏',
    'P2506': '披露不实(其它)',
    'P2507': '欺诈上市',
    'P2508': '出资违规',
    'P2509': '擅自改变资金用途',
    'P2510': '占用公司资产',
    'P2511': '内幕交易',
    'P2512': '违规买卖股票',
    'P2513': '操纵股价',
    'P2514': '违规担保',
    'P2515': '一般会计处理不当',
    'P2516': '偷税',
    'P2517': '逃避追缴欠税',
    'P2518': '骗取出口退税',
    'P2519': '抗税',
    'P2520': '虚开增值税专用发票或者虚开用于骗取出口退税、抵扣税款的其他发票',
    'P2521': '虚开普通发票',
    'P2522': '私自印制、伪造、变造发票，非法制造发票防伪专用品，伪造发票监制章',
    'P2523': '具有偷税、逃避追缴欠税、骗取出口退税、抗税、虚开发票等行为，经税务机关检查确认走逃（失联）',
    'P2524': '未缴或少缴税款(欠税)',
    'P2599': '其他'
}

vio_type_pct = []
vio_summary_tmp = vio_summary.copy()
vio_type = vio_summary_tmp['违规类型'].apply(lambda x: re.split(r'[ ;、,]', x))
vio_type = vio_type.explode().unique()

vio_dataset = vio_summary['违规类型']

for i in vio_type:
    i_in_col = vio_dataset.apply(lambda col: i in col)
    count_i = i_in_col.sum()
    vio_type_pct.append(count_i/len(vio_dataset))

vio_type_pct_df = pd.DataFrame(vio_type_pct, columns=['Percentage'])
vio_type_pct_df['ViolateType'] = vio_type.T
vio_type_pct_df = vio_type_pct_df.sort_values(by='Percentage', ascending=False)
vio_type_pct_df['ViolateType'] = vio_type_pct_df['ViolateType'].replace(violation_dict)
# vio_type_pct_df.to_csv('count_fraud_types.csv')


with pd.ExcelWriter('./output/fraud_data.xlsx') as writer:
    vio_num_ratio.to_excel(writer, sheet_name='annual fraud ratio', index=True)
    fraud_duration.to_excel(writer, sheet_name='fraud duration', index=False)
    vio_type_pct_df.to_excel(writer, sheet_name='fraud type pct', index=False)
    industries_fraud_pct.to_excel(writer, sheet_name='industries fraud pct', index=True)
    industries_fraud_count.to_excel(writer, sheet_name='industries fraud count', index=True)
    duration_each_stock_df_output.to_excel(writer, sheet_name='more than 10 y duration', index=False)

print('===============导出数据完成==================')
