import pandas as pd
import os
import glob
import sys


def read_industries_excel(this_file_pattern):
    files = glob.glob(this_file_pattern)

    this_industries = pd.DataFrame(columns=['证券代码', '年份', '所属行业'])
    for file in files:
        filename = os.path.basename(file)
        year = filename[8:12]
        tmp_years = pd.read_excel(file, sheet_name='万得')

        # 使用中信一级分类
        cols_to_keep = [col for col in tmp_years.columns if "所属中信行业指数代码" in col and "一级行业" in col]
        other_cols = ['证券代码', '证券简称']
        cols_to_keep = other_cols + cols_to_keep

        tmp_years = tmp_years[cols_to_keep]
        tmp_years.columns = other_cols + ['所属行业']
        this_industries = pd.concat([this_industries, tmp_years], axis=0)
        this_industries['年份'] = this_industries['年份'].fillna(int(year))

        print(f"Year: {year}, DataFrame shape: {tmp_years.shape[0]}")

    return this_industries

def main_dataset_obtain(folder_path):

    # ===========================import the data============================
    # 编码检测
    # with open(csv_files[0], 'rb') as f:
    #     encoding_type = chardet.detect(f.read())
    print('===============组合多个数据库的异常样本结果=================')

    try:
        csmar_data = pd.read_csv('./output/违规案例整理_csmar.csv', header=0, index_col=0)
        print('===============读取：违规案例整理_csmar.csv=================')
    except FileNotFoundError:
        print('No such file or directory: 违规案例整理_csmar.csv')
        csmar_data = pd.DataFrame()

    try:
        wind_data = pd.read_csv('./output/违规案例整理_wind_ai.csv', header=0, index_col=0)
        print('===============读取：违规案例整理_wind_ai.csv=================')
        wind_data.columns = ['证券代码', '年份']
    except FileNotFoundError:
        print('No such file or directory: 违规案例整理_wind_ai.csv')
        wind_data = pd.DataFrame()

    all_violate_dataset = pd.merge(left=csmar_data, right=wind_data, on=['证券代码', '年份'], how='outer')
    all_violate_dataset['违规类型'] = all_violate_dataset['违规类型'].fillna('wind:信息披露虚假或严重误导性陈述')

    print('==============正在导入行业表数据==================')
    this_folder_path = folder_path + '\\股票池和行业分类'
    file_pattern = os.path.join(this_folder_path, 'industry????12.xlsx')
    years = read_industries_excel(file_pattern)

    # 上市第一年所属行业分类缺失，后面年份有，使用后一年的数据填充
    years = years.sort_values(by=['证券代码', '年份'])
    years['所属行业'] = years.groupby('证券代码')['所属行业'].bfill()
    # 某些股票当年退市，行业表不包含该股票数据，则回溯一年数据填充
    violate_cases = years.copy()
    violate_cases = pd.merge(all_violate_dataset, violate_cases, on=['证券代码', '年份'], how='left')
    violate_cases['所属行业'] = violate_cases.groupby('证券代码')['所属行业'].ffill()
    violate_cases['证券简称'] = violate_cases.groupby('证券代码')['证券简称'].ffill()
    violate_cases['label'] = 1

    # 添加控制数据集
    all_dataset = pd.merge(years, violate_cases, on=['证券代码', '年份'], how='left', suffixes=('_years', ''))
    all_dataset['label'] = all_dataset['label'].fillna(0)
    all_dataset['所属行业'] = all_dataset['所属行业'].fillna(all_dataset['所属行业_years'])
    all_dataset['证券简称'] = all_dataset['证券简称'].fillna(all_dataset['证券简称_years'])
    all_dataset = all_dataset.drop(columns=['所属行业_years', '证券简称_years'])

    # all_dataset = all_dataset.loc[
    #     (all_dataset['年份'] >= start_year) & (all_dataset['年份'] <= end_year)].copy()

    all_dataset = all_dataset.reset_index(drop=True)
    all_dataset.to_excel(r'./output/违规案例整理.xlsx')
    print('==========已导出./output/违规案例整理.xlsx=============')
    return all_dataset


if __name__ == '__main__':
    folder_path = 'D:/数据库'                        # sys.argv[1]
    main_dataset_obtain(sys.argv[1])