import pandas as pd
import sys


def add_ex_suffix(code):
    '''
    添加交易所后缀，统一命名方式
    :param code: 不带后缀的6位数字
    :return: 带后缀的6位数字，str
    '''
    code = str(code)
    exchange = code[:3]
    if exchange in ('600', '601', '603', '688', '605', '900'):
        return code + '.SH'
    elif exchange in ('000', '001', '002', '003', '300', '301', '200'):
        return code + '.SZ'
    elif exchange[:2] in ('43', '83', '87', '92'):
        return code + '.BJ'
    else:
        print(code)
        raise ValueError('New types of stock_code.')


def main_dataset_obtain(violate_sheet_name, folder_path):

    # ===========================import the data============================
    # 编码检测
    # with open(csv_files[0], 'rb') as f:
    #     encoding_type = chardet.detect(f.read())


    print('==============正在处理csmar违规表数据==================')
    df = pd.read_excel('./input/' + violate_sheet_name, header=0)
    df = df.iloc[2:]
    print(df.head())

    df['Symbol'] = df['Symbol'].apply(add_ex_suffix)

    # =================================violate data=======================================
    # 提取虚假申报案例
    print('================提取虚假申报案例===================')
    fal_dlr = df[df['ViolationTypeID'].str.contains(r'P2501|P2502|P2503|P2505|P2506')]
    fal_dlr = fal_dlr.loc[fal_dlr['IsViolated'] == 'Y'].copy()
    fal_dlr['ViolationTypeID'] = fal_dlr['ViolationTypeID'].apply(lambda col: col.split(';'))
    fal_dlr['ViolationYear'] = fal_dlr['ViolationYear'].str.split(';')

    # 用 fillna 替换 NaN 为空列表
    fal_dlr['ViolationTypeID'] = fal_dlr['ViolationTypeID'].apply(lambda x: x if isinstance(x, list) else [])
    fal_dlr['ViolationYear'] = fal_dlr['ViolationYear'].apply(lambda x: x if isinstance(x, list) else [])

    # 忽略没有违规年份为NaN的案例
    mismatch_index = fal_dlr[fal_dlr['ViolationTypeID'].apply(len) != fal_dlr['ViolationYear'].apply(len)].index
    filtered_fal_dlr = fal_dlr.drop(mismatch_index)

    # 数据透视
    fal_dlr_exploded = filtered_fal_dlr.explode(['ViolationTypeID', 'ViolationYear'], ignore_index=True)
    fal_dlr_exploded = fal_dlr_exploded[fal_dlr_exploded['ViolationTypeID'].str.contains(r'P2501|P2502|P2503')]

    fal_dlr_exploded = fal_dlr_exploded[['Symbol', 'ViolationTypeID', 'ViolationYear']].copy()

    fal_dlr_pivot = fal_dlr_exploded.pivot_table(index=['Symbol', 'ViolationYear'], values='ViolationTypeID',
                                                 aggfunc=lambda x: ', '.join(x))
    fal_dlr_pivot = fal_dlr_pivot.reset_index()
    fal_dlr_pivot.columns = ['证券代码', '年份', '违规类型']
    # 忽略违规年份为N/A的数据
    fal_dlr_pivot = fal_dlr_pivot.loc[fal_dlr_pivot['年份'] != 'N/A']
    fal_dlr_pivot['年份'] = fal_dlr_pivot['年份'].astype(int)

    fal_dlr_pivot.to_csv(r'./output/违规案例整理_csmar.csv')
    print('==========已导出./output/违规案例整理_csmar.csv=============')
    return fal_dlr_pivot


if __name__ == '__main__':
    violate_sheet_name = 'STK_Violation_Main.xlsx'  # sys.argv[1]
    folder_path = 'D:/数据库'                        # sys.argv[2]
    main_dataset_obtain(sys.argv[1], sys.argv[2])