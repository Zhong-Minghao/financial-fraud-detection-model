import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from fuzzywuzzy import fuzz
import sys
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def read_data_of_all_years(folder_path, sheet_folder_path, this_years, encode='gbk'):
    tmp_sht = {}
    for y in this_years:
        tmp_sht[y] = pd.read_csv(folder_path + sheet_folder_path + str(y) + '1231.csv', encoding=encode)
    return tmp_sht


def read_data_of_last_d_every_months(yearlist: list, all_months_path: dict, encode = 'gbk') -> dict:
    tmp_sht = {}
    for m in all_months_path:
        year = m[0:-3]
        this_csv = pd.read_csv(all_months_path[m], encoding=encode)
        if int(year) in yearlist:
            if year in tmp_sht:
                tmp_sht[year].append(this_csv)
            else:
                tmp_sht[year] = [this_csv]
    return tmp_sht


def get_file_in_last_day_every_month(file_pattern):
    files = glob.glob(file_pattern)

    # 只有命名的空A股日收益率csv文件大小为2
    non_empty_files = [file for file in files if os.path.getsize(file) > 5]

    # 提取日期并排序
    file_info = []
    for file in non_empty_files:
        # 提取文件名中的日期部分
        file_name = os.path.basename(file)
        date_str = file_name[-12:-4]
        date = datetime.strptime(date_str, "%Y%m%d")
        file_info.append((file, date))

    file_info.sort(key=lambda x: x[1])

    last_files_per_month = {}
    for file, date in file_info:
        year_month = date.strftime("%Y-%m")
        last_files_per_month[year_month] = file

    return last_files_per_month


def get_features_from_sht(sht_name, year, stock_list, features_to_take):
    sheets = {
        'income stmt': income_stmt,
        'balance sht': balance_sht,
        'indicator sht': indicator_sht,
        'cashflow sht': cashflow_sht
    }

    if sht_name not in sheets:
        raise ValueError('sht_name should be one of the follows: income stmt, balance sht, indicator sht, cashflow sht.')

    this_sht = sheets[sht_name][year]

    this_sht = this_sht.loc[this_sht['S_INFO_WINDCODE'].isin(stock_list), ['S_INFO_WINDCODE', 'ANN_DT'] + features_to_take]
    # 仅保留第一次公布的数值
    this_sht = this_sht.loc[this_sht.groupby('S_INFO_WINDCODE')['ANN_DT'].idxmin()]
    this_sht = this_sht.drop(columns=['ANN_DT'])

    return this_sht


def annualized_features(annual_market_data: dict, columns_to_take: str):
    """
    读取月度数据，返回年度均值与标准差
    :param annual_market_data:
    :param columns_to_take:
    :return:
    """
    tmp_sht = {}
    for this_year in annual_market_data:
        this_df = annual_market_data[this_year][0]
        this_df = this_df[['S_INFO_WINDCODE', columns_to_take]].copy()
        annualized_feature = this_df
        for this_df in annual_market_data[this_year][1:]:
            this_df = this_df[['S_INFO_WINDCODE', columns_to_take]].copy()
            annualized_feature = pd.merge(annualized_feature, this_df, on='S_INFO_WINDCODE', how='outer', suffixes=('', '_dup'))
        annualized_feature[columns_to_take + '_mean'] = annualized_feature.iloc[:, 1:].mean(axis=1)
        annualized_feature[columns_to_take + '_std'] = annualized_feature.iloc[:, 1:].std(axis=1)
        result = annualized_feature[['S_INFO_WINDCODE', columns_to_take + '_mean', columns_to_take + '_std']]
        tmp_sht[int(this_year)] = result
    return tmp_sht


def read_csv_with_handling(file_path):
    try:
        return pd.read_csv(file_path, encoding='gbk')
    except pd.errors.EmptyDataError:
        # print(f"File {file_path} is empty.")
        return pd.DataFrame()
    except FileNotFoundError:
        # print(f"File {file_path} does not exist.")
        return pd.DataFrame()


# ==============================特征构建====================================
def compare_names(name1, name2, threshold=60):
    # 将NaN值替换为空字符串
    name1 = str(name1) if pd.notna(name1) else ''
    name2 = str(name2) if pd.notna(name2) else ''
    similarity = fuzz.ratio(name1, name2)
    return similarity >= threshold


def get_data_this_year(current_year, current_stock_pool):

    # =================全指标跑一遍===============
    columns_to_drop = ['S_INFO_WINDCODE', 'ANN_DT', 'WIND_CODE', 'REPORT_PERIOD', 'STATEMENT_TYPE', 'CRNCY_CODE',
                       'S_INFO_COMPCODE', 'OPDATE', 'OPMODE', 'S_INFO_COMPCODE', 'MEMO', 'ACTUAL_ANN_DT',
                       'IS_CALCULATION']
    balance_sht_features = [i for i in balance_sht[yearlist[0]].columns if i not in columns_to_drop]
    income_stmt_features = [i for i in income_stmt[yearlist[0]].columns if i not in columns_to_drop]
    indicator_sht_features = [i for i in indicator_sht[yearlist[0]].columns if i not in columns_to_drop]
    cashflow_sht_features = [i for i in cashflow_sht[yearlist[0]].columns if i not in columns_to_drop]
    # ===========================================

    this_bs = get_features_from_sht('balance sht', current_year, current_stock_pool, balance_sht_features)
    this_it = get_features_from_sht('income stmt', current_year, current_stock_pool, income_stmt_features)
    this_id = get_features_from_sht('indicator sht', current_year, current_stock_pool, indicator_sht_features)
    this_cf = get_features_from_sht('cashflow sht', current_year, current_stock_pool, cashflow_sht_features)

    shift_bs = get_features_from_sht('balance sht', current_year - 1, current_stock_pool, balance_sht_features)
    shift_it = get_features_from_sht('income stmt', current_year - 1, current_stock_pool, income_stmt_features)
    shift_id = get_features_from_sht('indicator sht', current_year - 1, current_stock_pool, indicator_sht_features)
    shift_cf = get_features_from_sht('cashflow sht', current_year - 1, current_stock_pool, cashflow_sht_features)

    # 合并为一张表，对齐索引【S_INFO_WINDCODE】，去年项统一命名在最开始加一个shift_
    shift_bs.columns = [shift_bs.columns[0]] + ('shift_' + shift_bs.columns[1:]).tolist()
    shift_it.columns = [shift_it.columns[0]] + ('shift_' + shift_it.columns[1:]).tolist()
    shift_id.columns = [shift_id.columns[0]] + ('shift_' + shift_id.columns[1:]).tolist()
    shift_cf.columns = [shift_cf.columns[0]] + ('shift_' + shift_cf.columns[1:]).tolist()

    this_dfs = [this_bs, this_it, this_id, this_cf, shift_bs, shift_it, shift_id, shift_cf]
    this_combine = this_dfs[0]
    for df in this_dfs[1:]:
        this_combine = pd.merge(this_combine, df, on='S_INFO_WINDCODE', how='outer', suffixes=('', '_dup'))

    this_combine = this_combine[[col for col in this_combine.columns if '_dup' not in col]]

    # 前十大股东数据表提取字段
    shareholder_sht_features = ['S_INFO_WINDCODE', 'S_HOLDER_HOLDERCATEGORY', 'S_HOLDER_NAME', 'S_HOLDER_QUANTITY',
                                'S_HOLDER_PCT']
    this_sh = shareholder_sht[current_year]
    this_sh = this_sh.loc[this_sh['S_INFO_WINDCODE'].isin(current_stock_pool), shareholder_sht_features]

    shift_shareholder_sht_features = ['S_INFO_WINDCODE', 'S_HOLDER_HOLDERCATEGORY', 'S_HOLDER_NAME',
                                      'S_HOLDER_QUANTITY',
                                      'S_HOLDER_PCT']
    shift_sh = shareholder_sht[current_year - 1]
    shift_sh = shift_sh.loc[shift_sh['S_INFO_WINDCODE'].isin(current_stock_pool), shift_shareholder_sht_features]
    shift_sh.columns = [shift_sh.columns[0]] + ('shift_' + shift_sh.columns[1:]).tolist()

    this_mg = manager_sht[current_year]
    this_mg = this_mg.loc[
        this_mg['S_INFO_WINDCODE'].isin(current_stock_pool), ['S_INFO_WINDCODE', 'S_MANAGER_QUANTITY']]

    this_au = audit_sht[current_year]
    this_au = this_au.loc[this_au['S_INFO_WINDCODE'].isin(current_stock_pool)]

    shift_au = audit_sht[current_year - 1]
    shift_au = shift_au.loc[shift_au['S_INFO_WINDCODE'].isin(current_stock_pool)]

    this_bd = board.loc[(board['S_INFO_WINDCODE'].isin(current_stock_pool)) &
                        (board['S_INFO_MANAGER_STARTDATE'] < current_year * 10000 + 630) & (
                                board['S_INFO_MANAGER_LEAVEDATE'] > current_year * 10000 + 630)]

    shift_bd = board.loc[(board['S_INFO_WINDCODE'].isin(current_stock_pool)) &
                         (board['S_INFO_MANAGER_STARTDATE'] < (current_year - 1) * 10000 + 630) & (
                                 board['S_INFO_MANAGER_LEAVEDATE'] > (current_year - 1) * 10000 + 630)]

    this_mk = market[current_year]
    this_mk = this_mk.loc[this_mk['S_INFO_WINDCODE'].isin(current_stock_pool)]

    this_vd = valuation_derivative_sht[current_year]

    this_sf = staff_sht[current_year]
    shift_sf = staff_sht[current_year - 1]
    shift_sf.columns = [shift_sf.columns[0]] + ('shift_' + shift_sf.columns[1:]).tolist()

    this_rpt = rpt_sht[current_year]

    shift_rpt = rpt_sht[current_year-1]
    shift_rpt.columns = [shift_rpt.columns[0]] + ('shift_' + shift_rpt.columns[1:]).tolist()

    return this_combine, this_sh, shift_sh, this_mg, this_au, shift_au, this_bd, shift_bd, this_mk, this_vd, this_sf, shift_sf, this_rpt, shift_rpt


def feature_engineering(current_year, current_stock_pool):
    global this_combine
    this_combine, this_sh, shift_sh, this_mg, this_au, shift_au, this_bd, shift_bd, this_mk, this_vd, this_sf, shift_sf, this_rpt, shift_rpt = get_data_this_year(
        current_year, current_stock_pool)

    # ====================================财务指标=========================================
    tmp_features = this_combine[['S_INFO_WINDCODE']].copy()
    # tmp_features = this_combine.copy()
    bs_features = ['TRADABLE_FIN_ASSETS', 'NOTES_RCV', 'DVD_RCV', 'INT_RCV', 'CONSUMPTIVE_BIO_ASSETS', 'DEFERRED_EXP', 'NON_CUR_ASSETS_DUE_WITHIN_1Y', 'SETTLE_RSRV', 'LOANS_TO_OTH_BANKS', 'PREM_RCV', 'RCV_FROM_REINSURER', 'RCV_FROM_CEDED_INSUR_CONT_RSRV', 'RED_MONETARY_CAP_FOR_SALE', 'OTH_CUR_ASSETS', 'FIN_ASSETS_AVAIL_FOR_SALE', 'HELD_TO_MTY_INVEST', 'LONG_TERM_EQY_INVEST', 'INVEST_REAL_ESTATE', 'TIME_DEPOSITS', 'OTH_ASSETS', 'LONG_TERM_REC', 'FIX_ASSETS', 'CONST_IN_PROG', 'PROJ_MATL', 'FIX_ASSETS_DISP', 'PRODUCTIVE_BIO_ASSETS', 'OIL_AND_NATURAL_GAS_ASSETS', 'INTANG_ASSETS', 'R_AND_D_COSTS', 'GOODWILL', 'LONG_TERM_DEFERRED_EXP', 'DEFERRED_TAX_ASSETS', 'LOANS_AND_ADV_GRANTED', 'OTH_NON_CUR_ASSETS', 'CASH_DEPOSITS_CENTRAL_BANK', 'ASSET_DEP_OTH_BANKS_FIN_INST', 'PRECIOUS_METALS', 'DERIVATIVE_FIN_ASSETS', 'AGENCY_BUS_ASSETS', 'SUBR_REC', 'RCV_CEDED_UNEARNED_PREM_RSRV', 'RCV_CEDED_CLAIM_RSRV', 'RCV_CEDED_LIFE_INSUR_RSRV', 'RCV_CEDED_LT_HEALTH_INSUR_RSRV', 'MRGN_PAID', 'INSURED_PLEDGE_LOAN', 'CAP_MRGN_PAID', 'INDEPENDENT_ACCT_ASSETS', 'CLIENTS_CAP_DEPOSIT', 'CLIENTS_RSRV_SETTLE', 'INCL_SEAT_FEES_EXCHANGE', 'RCV_INVEST', 'ST_BORROW', 'BORROW_CENTRAL_BANK', 'DEPOSIT_RECEIVED_IB_DEPOSITS', 'LOANS_OTH_BANKS', 'TRADABLE_FIN_LIAB', 'NOTES_PAYABLE', 'ACCT_PAYABLE', 'ADV_FROM_CUST', 'FUND_SALES_FIN_ASSETS_RP', 'HANDLING_CHARGES_COMM_PAYABLE', 'EMPL_BEN_PAYABLE', 'TAXES_SURCHARGES_PAYABLE', 'INT_PAYABLE', 'DVD_PAYABLE', 'OTH_PAYABLE', 'ACC_EXP', 'DEFERRED_INC', 'ST_BONDS_PAYABLE', 'PAYABLE_TO_REINSURER', 'RSRV_INSUR_CONT', 'ACTING_TRADING_SEC', 'ACTING_UW_SEC', 'NON_CUR_LIAB_DUE_WITHIN_1Y', 'OTH_CUR_LIAB', 'LT_BORROW', 'BONDS_PAYABLE', 'LT_PAYABLE', 'SPECIFIC_ITEM_PAYABLE', 'PROVISIONS', 'DEFERRED_TAX_LIAB', 'DEFERRED_INC_NON_CUR_LIAB', 'OTH_NON_CUR_LIAB', 'LIAB_DEP_OTH_BANKS_FIN_INST', 'DERIVATIVE_FIN_LIAB', 'CUST_BANK_DEP', 'AGENCY_BUS_LIAB', 'OTH_LIAB', 'PREM_RECEIVED_ADV', 'DEPOSIT_RECEIVED', 'INSURED_DEPOSIT_INVEST', 'UNEARNED_PREM_RSRV', 'OUT_LOSS_RSRV', 'LIFE_INSUR_RSRV', 'LT_HEALTH_INSUR_V', 'INDEPENDENT_ACCT_LIAB', 'INCL_PLEDGE_LOAN', 'CLAIMS_PAYABLE', 'DVD_PAYABLE_INSURED', 'CAP_STK', 'CAP_RSRV', 'SPECIAL_RSRV', 'SURPLUS_RSRV', 'UNDISTRIBUTED_PROFIT', 'LESS_TSY_STK', 'PROV_NOM_RISKS', 'CNVD_DIFF_FOREIGN_CURR_STAT', 'UNCONFIRMED_INVEST_LOSS', 'MINORITY_INT']
    tmp_features = pd.concat([tmp_features, this_combine[bs_features].div(this_combine['TOT_ASSETS'], axis=0)], axis=1)

    cash_flow_features = ['CASH_RECP_DISP_WITHDRWL_INVEST', 'CASH_RECP_RETURN_INVEST', 'NET_CASH_RECP_DISP_FIOLTA', 'NET_CASH_RECP_DISP_SOBU', 'OTHER_CASH_RECP_RAL_INV_ACT', 'CASH_PAY_ACQ_CONST_FIOLTA', 'CASH_PAID_INVEST', 'NET_CASH_PAY_AQUIS_SOBU', 'OTHER_CASH_PAY_RAL_INV_ACT', 'NET_INCR_PLEDGE_LOAN', 'CASH_PREPAY_AMT_BORR', 'CASH_PAY_DIST_DPCP_INT_EXP', 'INCL_DVD_PROFIT_PAID_SC_MS', 'OTHER_CASH_PAY_RAL_FNC_ACT', 'CASH_RECP_CAP_CONTRIB', 'INCL_CASH_REC_SAIMS', 'CASH_RECP_BORROW', 'PROC_ISSUE_BONDS', 'OTHER_CASH_RECP_RAL_FNC_ACT', 'CASH_RECP_SG_AND_RS', 'RECP_TAX_RENDS', 'NET_INCR_DEP_COB', 'NET_INCR_LOANS_CENTRAL_BANK', 'NET_INCR_FUND_BORR_OFI', 'CASH_RECP_PREM_ORIG_INCO', 'NET_INCR_INSURED_DEP', 'NET_CASH_RECEIVED_REINSU_BUS', 'NET_INCR_DISP_TFA', 'NET_INCR_INT_HANDLING_CHRG', 'NET_INCR_DISP_FAAS', 'NET_INCR_LOANS_OTHER_BANK', 'NET_INCR_REPURCH_BUS_FUND', 'OTHER_CASH_RECP_RAL_OPER_ACT', 'CASH_PAY_GOODS_PURCH_SERV_REC', 'CASH_PAY_BEH_EMPL', 'PAY_ALL_TYP_TAX', 'NET_INCR_CLIENTS_LOAN_ADV', 'NET_INCR_DEP_CBOB', 'CASH_PAY_CLAIMS_ORIG_INCO', 'HANDLING_CHRG_PAID', 'COMM_INSUR_PLCY_PAID', 'OTHER_CASH_PAY_RAL_OPER_ACT', 'PLUS_PROV_DEPR_ASSETS', 'DEPR_FA_COGA_DPBA', 'AMORT_INTANG_ASSETS', 'LOSS_DISP_FIOLTA', 'LOSS_SCR_FA', 'LOSS_FV_CHG', 'INVEST_LOSS', 'DECR_DEFERRED_INC_TAX_ASSETS', 'DECR_INVENTORIES', 'DECR_OPER_PAYABLE', 'AMORT_LT_DEFERRED_EXP', 'DECR_DEFERRED_EXP', 'INCR_ACC_EXP', 'FIN_EXP', 'INCR_DEFERRED_INC_TAX_LIAB', 'INCR_OPER_PAYABLE']

    tmp_features = pd.concat([tmp_features, this_combine[cash_flow_features].div(this_combine['TOT_OPER_REV'], axis=0)], axis=1)

    income_stmt_features = ['INT_INC', 'NET_INT_INC', 'INSUR_PREM_UNEARNED', 'HANDLING_CHRG_COMM_INC', 'NET_HANDLING_CHRG_COMM_INC', 'NET_INC_OTHER_OPS', 'PLUS_NET_INC_OTHER_BUS', 'PREM_INC', 'LESS_CEDED_OUT_PREM', 'CHG_UNEARNED_PREM_RES', 'INCL_REINSURANCE_PREM_INC', 'NET_INC_SEC_TRADING_BROK_BUS', 'NET_INC_SEC_UW_BUS', 'NET_INC_EC_ASSET_MGMT_BUS', 'OTHER_BUS_INC', 'PLUS_NET_GAIN_CHG_FV', 'PLUS_NET_INVEST_INC', 'INCL_INC_INVEST_ASSOC_JV_ENTP', 'PLUS_NET_GAIN_FX_TRANS', 'LESS_OPER_COST', 'LESS_INT_EXP', 'LESS_HANDLING_CHRG_COMM_EXP', 'LESS_TAXES_SURCHARGES_OPS', 'LESS_IMPAIR_LOSS_ASSETS', 'PREPAY_SURR', 'TOT_CLAIM_EXP', 'CHG_INSUR_CONT_RSRV', 'DVD_EXP_INSURED', 'REINSURANCE_EXP', 'OPER_EXP', 'LESS_CLAIM_RECB_REINSURER', 'LESS_INS_RSRV_RECB_REINSURER', 'LESS_EXP_RECB_REINSURER', 'OTHER_BUS_COST']

    tmp_features = pd.concat([tmp_features, this_combine[income_stmt_features].div(this_combine['TOT_OPER_REV'], axis=0)], axis=1)

    # 一些直接从财务指标表得到的数据
    # 净资产同比, 每股留存收益， 每股盈余公积, 每股营业总收入, 息税折旧摊销前利润/负债合计
    # 流动资产/总资产, 经营活动产生的净流量增长率, 速动比率, ROE, 净资产收益率ROE(扣除非经常损益), 资产负债率, 营业收入净利润率,
    # 销售毛利率, 销售净利率, 总资产净利率, 存货周转率, 相对年初增长率-每股净资产(%)
    # 经营活动产生的现金流量净额/流动负债, 经营活动产生的现金流量净额/负债合计, 营业总收入同比增长率(%), 财务费用/营业总收入, 期末摊薄每股收益
    # 年化投入资本回报率, 总资产周转率, 应收账款周转率, 销售费用/营业总收入, 管理费用/营业总收入
    # 资产减值损失/营业总收入, 营业利润同比，所得税/利润总额, 资本支出／折旧和摊销, 固定资产周转率
    # 每股资本公积, 同比增长率-稀释每股收益(%), 相对年初增长率-归属母公司的股东权益(%), 每股未分配利润
    # 流动负债/负债合计, 息税前利润/营业总收入, 每股净资产, 利润总额／营业收入, 销售商品提供劳务收到的现金/营业收入
    # 每股息税前利润, 每股企业自由现金流量, 每股股东自由现金流量, 带息负债率（带息债务/全部投入资本）, 非营业利润／利润总额
    # 现金销售率（经营活动产生的现金流量净额/营业收入）, 归属于母公司的股东权益/带息债务， 流动比率
    columns_to_take = ['S_FA_YOY_EQUITY', 'S_FA_RETAINEDPS', 'S_FA_SURPLUSRESERVEPS', 'S_FA_GRPS', 'S_FA_EBITDATODEBT',
                       'S_FA_CATOASSETS', 'S_FA_YOYOCFPS', 'S_FA_QUICK', 'S_FA_ROE', 'S_FA_ROE_DEDUCTED', 'S_FA_DEBTTOASSETS', 'S_FA_PROFITTOGR',
                       'S_FA_GROSSPROFITMARGIN', 'S_FA_NETPROFITMARGIN', 'S_FA_ROA_YEARLY', 'S_FA_INVTURN', 'S_FA_YOYBPS',
                       'S_FA_OCFTOSHORTDEBT', 'S_FA_OCFTODEBT', 'S_FA_YOY_TR', 'S_FA_FINAEXPENSETOGR', 'S_FA_EPS_DILUTED2',
                       'S_FA_ROIC_YEARLY', 'S_FA_ASSETSTURN', 'S_FA_ARTURN', 'S_FA_SALEEXPENSETOGR', 'S_FA_ADMINEXPENSETOGR',
                       'S_FA_IMPAIRTOGR_TTM', 'S_FA_YOYOP', 'S_FA_TAXTOEBT', 'S_FA_CAPITALIZEDTODA', 'S_FA_FATURN',
                       'S_FA_SURPLUSCAPITALPS', 'S_FA_YOYEPS_DILUTED', 'S_FA_YOYEQUITY', 'S_FA_UNDISTRIBUTEDPS',
                       'S_FA_CURRENTDEBTTODEBT', 'S_FA_EBITTOGR', 'S_FA_BPS', 'S_FA_PROFITTOOP', 'S_FA_SALESCASHINTOOR',
                       'S_FA_EBITPS', 'S_FA_FCFFPS', 'S_FA_FCFEPS', 'S_FA_INTDEBTTOTOTALCAP', 'S_FA_NOPTOEBT',
                       'S_FA_OCFTOOR', 'S_FA_EQUITYTOINTERESTDEBT', 'S_FA_CURRENT']

    tmp_features[columns_to_take] = this_combine[columns_to_take].copy()

    # 三表中大额整数多于6个
    non_string_cols = this_combine.select_dtypes(exclude='object').columns
    mod_df = this_combine[non_string_cols].apply(lambda x: x % 1e6)
    tmp_features['Large_integer'] = ((mod_df == 0).sum(axis=1) > 6).astype(int)
    # 净利润:若年报净利润（含少数股东权益）为正，则该变量为 0，否则，则为 1。
    tmp_features['NET_PROFIT_INCL_MIN_INT_INC_NEGATIVE'] = (this_combine['NET_PROFIT_INCL_MIN_INT_INC'] <= 0).astype(int)
    # 去年净利润:若去年年报净利润（含少数股东权益）为正，则该变量为 0，否则，则为 1。
    tmp_features['shift_NET_PROFIT_INCL_MIN_INT_INC_NEGATIVE'] = (this_combine['shift_NET_PROFIT_INCL_MIN_INT_INC'] <= 0).astype(int)
    # 净利润增长率
    tmp_features['NET_PROFIT_INCL_MIN_INT_INC_YOY'] = this_combine['NET_PROFIT_INCL_MIN_INT_INC']/this_combine['shift_NET_PROFIT_INCL_MIN_INT_INC'] - 1
    # 货币资金增长率
    tmp_features['MONEY_CASH_YOY'] = this_combine['MONETARY_CAP']/this_combine['shift_MONETARY_CAP'] - 1
    # 存货占流动资产的比率
    tmp_features['INVTOCUR'] = this_combine['INVENTORIES'] / this_combine['TOT_CUR_ASSETS']
    # 应收账款占流动资产比例
    tmp_features['RECTOCUR'] = this_combine['ACCT_RCV'] / this_combine['TOT_CUR_ASSETS']
    # 预付款项占流动资产比例
    tmp_features['PREPAYTOCUR'] = this_combine['PREPAY'] / this_combine['TOT_CUR_ASSETS']
    # 应收款项占比: （应收账款+其他应收款）/营业收入
    tmp_features['RECTOREV'] = (this_combine['ACCT_RCV'] + this_combine['OTH_RCV']) / this_combine['TOT_OPER_REV']
    # 应收款项变化率
    tmp_shift = (this_combine['shift_ACCT_RCV'] + this_combine['shift_OTH_RCV']) / this_combine['shift_TOT_OPER_REV']
    tmp_features['RECTOREV_YOY'] = (tmp_features['RECTOREV'] / tmp_shift - 1).copy()
    # 应付账款周转率
    tmp_features['S_FA_APTURN'] = this_combine['TOT_OPER_REV'] / this_combine['ACCT_PAYABLE']
    # 应付款项占比:（应付账款+应付票据）/营业成本
    this_combine['NOTES_PAYABLE'] = this_combine['NOTES_PAYABLE'].fillna(0)
    tmp_features['PAYABLETOOPER_COST'] = (this_combine['ACCT_PAYABLE'] + this_combine['NOTES_PAYABLE']) / this_combine[
        'LESS_OPER_COST']
    # 应付变化率：应付款项占比YOY
    this_combine['shift_NOTES_PAYABLE'] = this_combine['shift_NOTES_PAYABLE'].fillna(0)
    tmp_shift = (this_combine['shift_ACCT_PAYABLE'] + this_combine['shift_NOTES_PAYABLE']) / this_combine[
        'shift_LESS_OPER_COST']
    tmp_features['PAYABLETOOPER_COST_YOY'] = (tmp_features['PAYABLETOOPER_COST'] / tmp_shift - 1).copy()
    # 存货周转率变化率
    tmp_features['S_FA_INVTURN_YOY'] = this_combine['S_FA_INVTURN'] / this_combine['shift_S_FA_INVTURN'] - 1
    # 流动资产增长率
    tmp_features['TOT_CUR_ASSETS_YOY'] = this_combine['TOT_CUR_ASSETS'] / this_combine['shift_TOT_CUR_ASSETS'] - 1
    # 流动负债增长率
    tmp_features['TOT_CUR_LIAB_YOY'] = this_combine['TOT_CUR_LIAB'] / this_combine['shift_TOT_CUR_LIAB'] - 1
    # 利息费用/营业总收入
    tmp_features['INTEREST_COST_TO_OPER_REV'] = this_combine['S_STMNOTE_FINEXP'] / this_combine['TOT_OPER_REV']
    # 无形资产变化率
    tmp_features['INTANGRATIO_YOY'] = (this_combine['INTANG_ASSETS'] / this_combine['shift_INTANG_ASSETS'] - 1).copy()
    # 固定资产/资产总计
    tmp_features['FIX_ASSETS_TO_TOT_ASSETS'] = this_combine['FIX_ASSETS'] / this_combine['TOT_ASSETS']
    # 固定资产变化率
    tmp_features['FIX_ASSETS_YOY'] = (this_combine['FIX_ASSETS'] / this_combine['shift_FIX_ASSETS'] - 1).copy()
    # 软资产比例：（资产总计-固定资产-在建工程-工程物资-货币资金）/资产总计
    # 工程物资,在建工程项目很多是nan认为正常，没有这项
    this_combine['PROJ_MATL'] = this_combine['PROJ_MATL'].fillna(0)
    this_combine['CONST_IN_PROG'] = this_combine['CONST_IN_PROG'].fillna(0)
    tmp_features['SOFTASSET'] = (this_combine['TOT_ASSETS'] - this_combine['FIX_ASSETS'] - this_combine[
        'CONST_IN_PROG'] - this_combine['PROJ_MATL'] - this_combine['MONETARY_CAP']) / this_combine['TOT_ASSETS']
    # 自由现金流变化率：股权自由现金流量(FCFE)YOY
    tmp_features['S_FA_FCFE_YOY'] = (this_combine['S_FA_FCFE'] / this_combine['shift_S_FA_FCFE'] - 1).copy()
    # 企业自由现金流变化率：企业自由现金流量(FCFF)YOY
    tmp_features['S_FA_FCFE_YOY'] = (this_combine['S_FA_FCFF'] / this_combine['shift_S_FA_FCFF'] - 1).copy()

    # 货币资金/流动资产
    tmp_features['MONETARYTOCUR'] = this_combine['MONETARY_CAP'] / this_combine['TOT_CUR_ASSETS']
    # 去年扣非ROE
    tmp_features['shift_S_FA_ROE_DEDUCTED'] = (this_combine['shift_S_FA_ROE_DEDUCTED'] < 0.06).astype(int)
    # 其他应收款增长率
    tmp_features['OTH_RCV_YOY'] = this_combine['OTH_RCV'] / this_combine['shift_OTH_RCV'] - 1
    # 研发支出/营业成本：研发费用(数据空值过多,手动加0)
    this_combine['RD_EXPENSE'] = this_combine['RD_EXPENSE'].fillna(0)
    tmp_features['RD_EXPENSE'] = this_combine['RD_EXPENSE'] / this_combine['LESS_OPER_COST']
    # 流动比:流动资产/流动负债
    tmp_features['CURRATIO'] = this_combine['TOT_CUR_ASSETS'] / this_combine['TOT_CUR_LIAB']
    # 营业外收入净利润率: 64营业外收支净额/利润总额
    tmp_features['S_FA_NONOPERATEPROFITTOEBT'] = this_combine['S_FA_NONOPERATEPROFITTOEBT']
    # 营业外支出净利润率
    tmp_features['S_FA_NONOPERATELOSSTOEBT'] = this_combine['LESS_NON_OPER_EXP'] / this_combine['TOT_PROFIT']
    # 资产减值损失/总资产
    tmp_features['IMPRLOSTOASSET'] = this_combine['LESS_IMPAIR_LOSS_ASSETS'] / this_combine['TOT_ASSETS']
    # 其他应收款/流动资产
    tmp_features['ORCVTOCUR'] = this_combine['OTH_RCV'] / this_combine['TOT_CUR_ASSETS']
    # 去年其他应收款/流动资产比例
    tmp_features['shift_ORCVTOCUR'] = this_combine['shift_OTH_RCV'] / this_combine['shift_TOT_CUR_ASSETS']
    # 预付款项变化率
    tmp_features['PREPAY_YOY'] = this_combine['PREPAY'] / this_combine['shift_PREPAY'] - 1
    # 营业利润现金净含量（盈余现金保障倍数）
    tmp_features['CASHCOVERAGERATIO'] = this_combine['NET_CASH_FLOWS_OPER_ACT'] / this_combine[
        'NET_PROFIT_INCL_MIN_INT_INC']
    # 无息流动负债/流动负债
    tmp_features['EXINTERESTDEBT_CURRENT_TO_TOT_CUR_LIAB'] = this_combine['S_FA_EXINTERESTDEBT_CURRENT'] / this_combine['TOT_CUR_LIAB']
    # 应交税费率
    tmp_features['TAXRATIO'] = this_combine['TAXES_SURCHARGES_PAYABLE'] / this_combine['TOT_OPER_REV']

    # 非经常性损益是否使净利润变号
    tmp_shift = pd.DataFrame()
    tmp_shift['A'] = this_combine['NET_PROFIT_INCL_MIN_INT_INC'] - this_combine['S_FA_EXTRAORDINARY']
    tmp_shift['B'] = this_combine['NET_PROFIT_INCL_MIN_INT_INC']
    tmp_shift = (tmp_shift > 0).astype(bool)
    tmp_shift['NET_PROFIT_MINUS_EXTRAORDINARY'] = tmp_shift['A'] ^ tmp_shift['B']
    tmp_features['NET_PROFIT_MINUS_EXTRAORDINARY'] = (tmp_shift['NET_PROFIT_MINUS_EXTRAORDINARY']).astype(int)

    # 非经常性损益/营业总收入
    tmp_features['S_FA_EXTRAORDINARY_TO_OPER_REV'] = this_combine['S_FA_EXTRAORDINARY'] / this_combine['TOT_OPER_REV']

    # 递延所得税负债/资产
    tmp_features['DEFERRED_TAX_LIAB_TO_TOT_ASSETS'] = this_combine['DEFERRED_TAX_LIAB'] / this_combine['TOT_ASSETS']

    # 应收票据造假: 应收票据/流动资产
    tmp_features['NOTETOCUR'] = this_combine['NOTES_RCV'] / this_combine['TOT_CUR_ASSETS']
    # 收到其他与经营活动有关的现金/应收票据
    tmp_features['OTHERCASHTONOTE'] = this_combine['OTHER_CASH_RECP_RAL_OPER_ACT']/this_combine['NOTES_RCV']
    # YOY(收到其他与经营活动有关的现金/应收票据)
    tmp_features['OTHERCASHTONOTE_YOY'] = (this_combine['OTHER_CASH_RECP_RAL_OPER_ACT'] * this_combine['shift_NOTES_RCV'] /
                                       this_combine['shift_OTHER_CASH_RECP_RAL_OPER_ACT'] / this_combine['NOTES_RCV'])

    # 应收账款增长率
    tmp_features['ACCT_RCV_YOY'] = this_combine['ACCT_RCV'] / this_combine['shift_ACCT_RCV'] - 1
    # YOY(应收账款)/ YOY(营业收入)
    tmp_features['ACCT_RCV_YOY_TO_S_FA_YOY'] = this_combine['ACCT_RCV'] / this_combine['shift_ACCT_RCV'] / this_combine['TOT_OPER_REV'] * this_combine['shift_TOT_OPER_REV']

    # 其他流动资产异常：其他流动资产/流动资产
    tmp_features['OTH_CURTOCUR'] = this_combine['OTH_CUR_ASSETS']/this_combine['TOT_CUR_ASSETS']
    # 其他流动资产异常：其他流动资产 / 流动资产 YOY
    tmp_shift = this_combine['shift_OTH_CUR_ASSETS']/this_combine['shift_TOT_CUR_ASSETS']
    tmp_features['OTH_CURTOCUR_YOY'] = (tmp_features['OTH_CURTOCUR'] / tmp_shift - 1).copy()

    # 分配股利、利润或偿付利息支付的现金/筹资活动现金流出小计
    tmp_features['CASH_PAY_DIST_DPCP_INT_EXP_TO_STOT_CASH_OUTFLOWS_FNC_ACT'] = this_combine['CASH_PAY_DIST_DPCP_INT_EXP'] / this_combine[
        'STOT_CASH_OUTFLOWS_FNC_ACT']

    # 资本公积增长率
    tmp_features['CAP_RSRV_YOY'] = (this_combine['CAP_RSRV'] / this_combine['shift_CAP_RSRV']) - 1

    # 收到其他与经营活动有关的现金/经营活动现金流入小计
    tmp_features['OTHER_CASH_TO_TOT_CASH_INFLOWS'] = this_combine['OTHER_CASH_RECP_RAL_OPER_ACT'] / this_combine['STOT_CASH_INFLOWS_OPER_ACT']
    # 收到其他与经营活动有关的现金增长率
    tmp_features['OTHER_CASH_RECP_RAL_OPER_ACT_YOY'] = this_combine['OTHER_CASH_RECP_RAL_OPER_ACT'] / this_combine['shift_OTHER_CASH_RECP_RAL_OPER_ACT'] - 1

    # 支付其他与经营活动有关的现金/经营活动现金流出小计
    tmp_features['OTHER_CASH_TO_TOT_CASH_OUTFLOWS'] = this_combine['OTHER_CASH_PAY_RAL_OPER_ACT'] / this_combine[
        'STOT_CASH_OUTFLOWS_OPER_ACT']
    # 支付其他与经营活动有关的现金增长率
    tmp_features['OTHER_CASH_PAY_RAL_OPER_ACT_YOY'] = this_combine['OTHER_CASH_PAY_RAL_OPER_ACT'] / this_combine[
        'shift_OTHER_CASH_PAY_RAL_OPER_ACT'] - 1

    # CASH_PAY_ACQ_CONST_FIOLTA_TO_CASH_PAID_INVEST
    tmp_features['CASH_PAY_INV_ACT_TO_TOT_CASH_OUTFLOWS'] = this_combine['CASH_PAY_ACQ_CONST_FIOLTA'] / this_combine[
        'STOT_CASH_OUTFLOWS_INV_ACT']

    # 投资活动现金流入/现金总流入
    tmp_features['STOT_CASH_INFLOWS_INV_ACT_RATIO'] = (this_combine['STOT_CASH_INFLOWS_INV_ACT'] /
                (this_combine['STOT_CASH_INFLOWS_INV_ACT'] + this_combine['NET_CASH_FLOWS_FNC_ACT'] + this_combine['STOT_CASH_INFLOWS_OPER_ACT']))

    # 购建固定资产、无形资产和其他长期资产支付的现金/投资支付的现金
    tmp_features['CASH_PAY_ACQ_CONST_FIOLTA_TO_CASH_PAID_INVEST'] = this_combine['CASH_PAY_ACQ_CONST_FIOLTA'] / this_combine[
        'CASH_PAID_INVEST']

    # 现金及现金等价物净增加额 / 期初现金及现金等价物余额
    tmp_features['NET_INCR_CASH_TO_CASH_EQU_BEG_PERIOD'] = this_combine['NET_INCR_CASH_CASH_EQU'] / this_combine['CASH_CASH_EQU_BEG_PERIOD']

    # ======================持股信息========================
    # 前五大持股股东持股比例
    tmp_top5 = pd.DataFrame(this_sh.groupby('S_INFO_WINDCODE')['S_HOLDER_PCT'].apply(lambda x: x.nlargest(5).sum()).reset_index())
    this_combine = pd.merge(this_combine, tmp_top5, on='S_INFO_WINDCODE', how='left')
    tmp_features['TOP5SHAREHOLDER_PCT'] = this_combine['S_HOLDER_PCT']

    # 股权集中度: 第一大股东的持股比例/第二和第三大股东的持股比例之和
    def max_diff(group):
        top_three = group['S_HOLDER_PCT'].nlargest(3)  # 获取前三个最大的值
        if len(top_three) == 2:
            return top_three.iloc[0] / top_three.iloc[1]
        elif len(top_three) < 2:
            return 0
        return top_three.iloc[0] / (top_three.iloc[1] + top_three.iloc[2])

    tmp_top5 = pd.DataFrame(this_sh.groupby('S_INFO_WINDCODE').apply(max_diff),columns=['Equity Concentration'])
    this_combine = pd.merge(this_combine, tmp_top5, on='S_INFO_WINDCODE', how='left')
    tmp_features['Equity Concentration'] = this_combine['Equity Concentration'].copy()

    # 机构投资者持股比例:没数据，只能给出前10股东数据里面，机构持股比例, 同时只能靠名字猜测是否为机构投资者(???同一只股票，持股数量/持股比例不一样)
    institute_keywords = ['基金', 'Fund', 'FUND', '财富', 'Fortune', 'FORTUNE', '保险', 'Insurance', 'INSURANCE',
                          '信托', 'Trust', 'TRUST', '投资', 'Investment', 'INVESTMENT', '证券', 'Securities', 'SECURITIES',
                          '资本', 'Capital', 'CAPITAL']
    pattern = '|'.join(institute_keywords)

    this_institute_investor = this_sh.loc[this_sh['S_HOLDER_HOLDERCATEGORY'] == 2].copy()
    this_institute_investor = this_institute_investor.loc[this_institute_investor['S_HOLDER_NAME'].str.contains(pattern, case=False, na=False)]
    this_institute_investor = this_institute_investor.groupby('S_INFO_WINDCODE')['S_HOLDER_PCT'].apply(lambda x: x.nlargest(10).sum()).reset_index()
    this_institute_investor.columns = ['S_INFO_WINDCODE', 'INTT_IVSTR_PCT']

    this_combine = pd.merge(this_combine, this_institute_investor, on='S_INFO_WINDCODE',how='left')

    tmp_features['INTT_IVSTR_PCT'] = this_combine['INTT_IVSTR_PCT']
    del this_institute_investor

    # 机构投资者是否减持
    shift_institute_investor = shift_sh.loc[shift_sh['shift_S_HOLDER_HOLDERCATEGORY'] == 2].copy()
    shift_institute_investor = shift_institute_investor.loc[
        shift_institute_investor['shift_S_HOLDER_NAME'].str.contains(pattern, case=False, na=False)]
    shift_institute_investor = shift_institute_investor.groupby('S_INFO_WINDCODE')['shift_S_HOLDER_PCT'].apply(
        lambda x: x.nlargest(10).sum()).reset_index()
    shift_institute_investor.columns = ['S_INFO_WINDCODE', 'shift_INTT_IVSTR_PCT']
    this_combine = pd.merge(this_combine, shift_institute_investor, on='S_INFO_WINDCODE',
                            how='left')
    tmp_features['INTT_IVSTR_SELL'] = ((this_combine['INTT_IVSTR_PCT'] - this_combine['shift_INTT_IVSTR_PCT'])<0).astype(int)

    # 公司管理层持股数量
    tmp_shift = this_mg.groupby('S_INFO_WINDCODE')['S_MANAGER_QUANTITY'].sum().reset_index()
    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features['MAGEMENTSHARES_PCT'] = this_combine['S_MANAGER_QUANTITY']

    # =======================企业管理===========================
    # 审计意见（独热变量）
    categories = ['405001000', '405002000', '405010000', '405003000', '405004000', '405005000']
    tmp_shift = this_au.loc[this_au.groupby('S_INFO_WINDCODE')['ANN_DT'].idxmin()]
    tmp_shift = tmp_shift.drop_duplicates()
    tmp_shift = tmp_shift.drop(columns=['ANN_DT'])
    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features_dummies = tmp_features.copy()
    tmp_features_dummies['AUDIT_CATEGORY'] = this_combine['S_STMNOTE_AUDIT_CATEGORY'].copy()

    tmp_features_dummies = pd.get_dummies(tmp_features_dummies, columns=['AUDIT_CATEGORY'], drop_first=False)
    expected_columns = [f'AUDIT_CATEGORY_{cat}' for cat in categories[1:]]
    # 使用reindex来扩展列，填充缺失的列为0, 舍弃'405001000'避免多重共线性
    tmp_features_dummies = tmp_features_dummies.reindex(columns=expected_columns, fill_value=0)
    tmp_features = tmp_features.join(tmp_features_dummies)

    # 高管人数
    tmp_shift = this_bd.groupby('S_INFO_WINDCODE')['S_INFO_MANAGER_LEAVEDATE'].count().reset_index()
    tmp_shift.columns = ['S_INFO_WINDCODE', 'EXECUTIVE_COUNT']
    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features['EXECUTIVE_COUNT'] = this_combine['EXECUTIVE_COUNT']

    # 高管人数是否发生变动超5人
    tmp_shift = shift_bd.groupby('S_INFO_WINDCODE')['S_INFO_MANAGER_LEAVEDATE'].count().reset_index()
    tmp_shift.columns = ['S_INFO_WINDCODE', 'shift_EXECUTIVE_COUNT']
    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features['EXECUTIVE_COUNT_change'] = (abs(this_combine['EXECUTIVE_COUNT'] - this_combine['shift_EXECUTIVE_COUNT'])>5).astype(int)

    # 董事会人数
    tmp_shift = this_bd.loc[this_bd['S_INFO_MANAGER_TYPE'] == 0].copy()
    tmp_shift = tmp_shift.groupby('S_INFO_WINDCODE')['S_INFO_MANAGER_LEAVEDATE'].count().reset_index()
    tmp_shift.columns = ['S_INFO_WINDCODE', 'BOARD_COUNT']
    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features['BOARD_COUNT'] = this_combine['BOARD_COUNT']

    # 是否更换审计事务所(!!!!不准确！！！！！少了个‘有限公司’，或者同一间换了名字)
    tmp_shift = pd.merge(this_au, shift_au, on='S_INFO_WINDCODE', how='outer')
    tmp_shift['CHANGEAUDIT'] = tmp_shift.apply(lambda row: compare_names(row['S_STMNOTE_AUDIT_AGENCY_x'], row['S_STMNOTE_AUDIT_AGENCY_y']), axis=1)
    tmp_shift['CHANGEAUDIT'] = (~tmp_shift['CHANGEAUDIT']).astype(int)
    # tmp_shift['CHANGEAUDIT'] = (tmp_shift['S_STMNOTE_AUDIT_AGENCY_x'] != tmp_shift['S_STMNOTE_AUDIT_AGENCY_y']).astype(int)
    tmp_shift = tmp_shift[['S_INFO_WINDCODE', 'CHANGEAUDIT']]

    this_combine = pd.merge(this_combine, tmp_shift, on='S_INFO_WINDCODE', how='left')
    tmp_features['CHANGEAUDIT'] = this_combine['CHANGEAUDIT']

    # 是否再融资：股本+资本公积+有息负债的增加值，哑变量
    tmp_shift = (this_combine['CAP_STK'] + this_combine['CAP_RSRV'] + this_combine['S_FA_INTERESTDEBT']
                 - this_combine['shift_CAP_STK'] - this_combine['shift_CAP_RSRV'] - this_combine[
                     'shift_S_FA_INTERESTDEBT'])
    tmp_features['REFUND'] = (tmp_shift > 0).astype(int)

    # 职工人数
    this_combine = this_combine.merge(this_sf, on='S_INFO_WINDCODE', how='left')
    tmp_features['TOT_STAFF_NUMBER'] = this_combine['TOT_STAFF_NUMBER'].copy()

    # 支付给职工以及为职工支付的现金/职工人数
    tmp_features['WAGE_PER_STAFF'] = this_combine['CASH_PAY_BEH_EMPL'] / this_combine['TOT_STAFF_NUMBER']

    # 职工人数是否减少
    this_combine = this_combine.merge(shift_sf, on='S_INFO_WINDCODE', how='left')
    tmp_features['TOT_STAFF_NUMBER_change'] = ((this_combine['TOT_STAFF_NUMBER'] - this_combine['shift_TOT_STAFF_NUMBER']) < 0).astype(int)

    # 关联交易额/营业总收入
    this_combine = pd.merge(this_combine, this_rpt, on='S_INFO_WINDCODE', how='left')
    this_combine['S_RELATEDTRADE_AMOUNT'] = this_combine['S_RELATEDTRADE_AMOUNT'].fillna(0).copy()
    tmp_features['RELATIVE_PARTY_TRANSACTION_RATIO'] = this_combine['S_RELATEDTRADE_AMOUNT'] / this_combine['TOT_OPER_REV']

    # 关联交易占比变化率
    this_combine = pd.merge(this_combine, shift_rpt, on='S_INFO_WINDCODE', how='left')
    this_combine['shift_S_RELATEDTRADE_AMOUNT'] = this_combine['shift_S_RELATEDTRADE_AMOUNT'].fillna(0).copy()
    tmp_shift = this_combine['shift_S_RELATEDTRADE_AMOUNT'] / this_combine['shift_TOT_OPER_REV']
    tmp_features['RELATIVE_PARTY_TRANSACTION_RATIO_YOY'] = tmp_features['RELATIVE_PARTY_TRANSACTION_RATIO'] / tmp_shift - 1

    # =========================市场信息==========================
    # 年度换手率标准差，均值
    this_combine = pd.merge(this_combine, this_mk, on='S_INFO_WINDCODE', how='left')
    tmp_features['TURNOVER_M_mean'] = this_combine['TURNOVER_M_mean']
    tmp_features['TURNOVER_M_std'] = this_combine['TURNOVER_M_std']

    this_combine = pd.merge(this_combine, this_vd, on='S_INFO_WINDCODE', how='left')
    # 总市值, 市盈率(PE,TTM), 市现率(PCF,经营现金流TTM), 市销率(PS,TTM), 市净率(PB), 股价/每股派息
    columns_to_take = ['S_VAL_MV', 'S_VAL_PE_TTM', 'S_VAL_PCF_OCFTTM', 'S_VAL_PS_TTM', 'S_VAL_PB_NEW', 'S_PRICE_DIV_DPS']
    tmp_features[columns_to_take] = this_combine[columns_to_take].copy()
    tmp_features['S_PRICE_DIV_DPS'] = (1 / tmp_features['S_PRICE_DIV_DPS']).copy()

    # ===================规则项========================
    # 规避退市：净利润(Y - 1) < 0  AND    YOY(营业总收入) > 0.2
    tmp_features['Avoid_delisting'] = np.where((tmp_features['NET_PROFIT_INCL_MIN_INT_INC_NEGATIVE'] == 0) &
                                               (tmp_features['S_FA_YOY_TR'] > 20), 1, 0)
    # 利息率异常：是否虚增货币资金  Rule3：实际当年利息率 - 计算利息率>1%
    tmp_features['Inflated_interest'] = this_combine['FIN_EXP_INT_INC']/this_combine['MONETARY_CAP']

    return tmp_features


# =========================去除异常数值, 求分位=========================
# IQR, 1倍IQR
def tail_process_n_scale_group(df, factor=1):
    Q1 = df.quantile(0.25, numeric_only=True)
    Q3 = df.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    this_columns = lower_bound.index.tolist()

    # 拖尾处理，将异常值缩到上下限内
    df_processed = pd.DataFrame(np.where(df[this_columns] < lower_bound, lower_bound, df[this_columns]),
                                columns=this_columns, index=df.index)
    df_processed = pd.DataFrame(np.where(df_processed[this_columns] > upper_bound, upper_bound, df_processed[this_columns]),
                                columns=this_columns, index=df.index)

    # ================分组求分位==============
    group = df_processed.copy()
    scaler = MinMaxScaler()
    numeric_cols = group.select_dtypes(include=['number']).columns
    group[numeric_cols] = scaler.fit_transform(group[numeric_cols])

    return group


def is_binary_column(series):
    unique_values = series.dropna().unique()
    return set(unique_values).issubset({0, 1})

def annual_feature_engineering(current_year):
    current_stock_pool = cases.loc[cases['年份'] == str(current_year)]
    current_stock_pool = current_stock_pool.rename(columns={'证券代码': 'S_INFO_WINDCODE'})

    this_features = feature_engineering(current_year, current_stock_pool['S_INFO_WINDCODE'])

    # 去除异常值
    this_features = this_features.replace([np.inf, -np.inf], np.nan)

    normalize_columns, other_columns = [], []

    for col in this_features.columns:
        if is_binary_column(this_features[col]):
            other_columns.append(col)
        else:
            normalize_columns.append(col)

    # 去极端值并标准化
    normalize_features = this_features[normalize_columns].copy()
    normalize_features = pd.merge(normalize_features, current_stock_pool[['S_INFO_WINDCODE', '所属行业']],
                                  on=['S_INFO_WINDCODE'], how='outer')
    num_normalize_features = normalize_features.drop(columns=['S_INFO_WINDCODE'])
    n_num_normalize_features = num_normalize_features.groupby('所属行业', group_keys=False).apply(tail_process_n_scale_group)
    other_columns = ['S_INFO_WINDCODE'] + other_columns
    normalize_features = pd.concat([this_features[other_columns], n_num_normalize_features], axis=1)
    new_columns = pd.DataFrame({'年份': [str(current_year)] * len(normalize_features)})
    normalize_features = pd.concat([normalize_features, new_columns], axis=1)
    return normalize_features


if __name__ == '__main__':

    # para_setting = pd.read_excel('./input/参数设置.xlsx', header=0)
    # start_year = para_setting.loc[para_setting['名称'] == 'start_year', '参数取值'].values[0]
    # end_year = para_setting.loc[para_setting['名称'] == 'end_year', '参数取值'].values[0]
    # folder_path = para_setting.loc[para_setting['名称'] == '通用数据路径', '参数取值'].values[0]
    # predict_year = para_setting.loc[para_setting['名称'] == 'predict_year', '参数取值'].values[0]

    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    predict_year = int(sys.argv[3])
    folder_path = sys.argv[4]
    # start_year = 2001
    # end_year = 2022
    # predict_year = 2023
    # folder_path = r'D:/数据库'

    assert start_year < end_year and predict_year > end_year

    # 待研究的年份列表
    yearlist = [i for i in range(start_year - 1, end_year + 1)] + [predict_year]
    # 训练集要保证能读到所有待研究年份的前一年（start=2001, 表格至少要到2000）（因为涉及到同比），同时附带读取测试集数据
    yearlist_full_data = [i for i in range(start_year - 1, end_year + 1)] + [predict_year - 1, predict_year]
    yearlist_full_data = sorted(list(set(yearlist_full_data)))

    print('==============正在导入违规案例数据==================')
    cases = pd.read_excel('./output/违规案例整理.xlsx', sheet_name='Sheet1', dtype=str, index_col=0)

    file_pattern = os.path.join(folder_path + "\股票池和行业分类", 'industry????12.xlsx')

    print('==========正在导入财务报表，公司股东等数据============')
    sheet_folder_path = '/wind财务报表'
    income_stmt = read_data_of_all_years(folder_path, sheet_folder_path + '/A股利润表', this_years=yearlist_full_data)
    indicator_sht = read_data_of_all_years(folder_path, sheet_folder_path + '/中国A股财务指标', this_years=yearlist_full_data)
    balance_sht = read_data_of_all_years(folder_path, sheet_folder_path + '/A股资产负债表', this_years=yearlist_full_data)
    cashflow_sht = read_data_of_all_years(folder_path, sheet_folder_path + '/A股现金流量表', this_years=yearlist_full_data)
    shareholder_sht = read_data_of_all_years(folder_path, '/wind中国A股股东数据/中国A股前十大股东', this_years=yearlist_full_data)
    manager_sht = read_data_of_all_years(folder_path, '/wind中国A股公司治理/中国A股公司管理层持股及报酬',
                                         this_years=yearlist_full_data[1:], encode='latin-1')

    # 审计数据不集中于6月，分散于全年
    # audit_sht_features = ['S_INFO_WINDCODE', 'ANN_DT', 'S_STMNOTE_AUDIT_CATEGORY', 'S_STMNOTE_AUDIT_AGENCY']
    dates = ['0331', '0630', '0930', '1231']
    audit_sht = {}
    for y in yearlist_full_data:
        read_dfs = []
        for date in dates:
            tmp_sht1 = read_csv_with_handling(folder_path + sheet_folder_path + f'\中国A股审计意见{y}{date}.csv')
            read_dfs.append(tmp_sht1)
        tmp_sht = pd.concat(read_dfs, axis=0)
        tmp_sht = tmp_sht.reset_index(drop=True)
        tmp_sht['ANN_DT'] = tmp_sht['ANN_DT'].fillna(50000000)
        audit_sht[y] = tmp_sht

    sheet_folder_path = '/wind中国A股公司治理'
    # 高管数据
    board = pd.read_csv(folder_path + sheet_folder_path + '/中国A股公司管理层成员.csv', usecols=[0, 2, 7, 8, 9],
                        encoding='GB18030')
    board['S_INFO_MANAGER_LEAVEDATE'] = board['S_INFO_MANAGER_LEAVEDATE'].fillna(np.inf)

    # 员工信息不集中于6月，分散于全年
    staff_sht = {}
    for y in yearlist_full_data:
        read_dfs = []
        for date in dates:
            tmp_sht1 = read_csv_with_handling(folder_path + sheet_folder_path + f'/中国A股员工构成{y}{date}.csv')
            read_dfs.append(tmp_sht1)
        tmp_sht = pd.concat(read_dfs, axis=0)
        if not tmp_sht.empty:
            tmp_sht['TOT_STAFF_NUMBER'] = round(tmp_sht['STAFF_NUMBER'].fillna(0) / tmp_sht['PROPORTION'] * 100)
            tmp_sht = tmp_sht.drop_duplicates(subset=['S_INFO_WINDCODE'], keep='first')
            tmp_sht = tmp_sht.reset_index(drop=True)
            tmp_sht = tmp_sht[['S_INFO_WINDCODE', 'TOT_STAFF_NUMBER']].copy()
        else:
            tmp_sht = pd.DataFrame(columns=['S_INFO_WINDCODE', 'TOT_STAFF_NUMBER'])

        staff_sht[y] = tmp_sht

    del read_dfs

    # 行情数据
    # 读取表格：日收益率
    file_pattern = os.path.join(folder_path + "\windA股行情衍生数据", 'A股日收益率????????.csv')
    last_files_per_month = get_file_in_last_day_every_month(file_pattern)
    annual_market_data = read_data_of_last_d_every_months(yearlist, last_files_per_month)
    market = annualized_features(annual_market_data, 'TURNOVER_M')

    valuation_derivative_sht = read_data_of_all_years(folder_path, r"\windA股行情衍生数据\A股日行情估值指标", this_years=yearlist)

    # 关联交易数据
    rpt_sht = {}
    for y in yearlist_full_data:
        file_pattern = os.path.join(folder_path, f'wind中国A股重大事件/中国A股关联交易{y}*.csv')
        files = glob.glob(file_pattern)
        filtered_files = [f for f in files if int(f.split('关联交易')[1][4:8]) <= 630]

        file_pattern = os.path.join(folder_path, f'wind中国A股重大事件/中国A股关联交易{y - 1}*.csv')
        files = glob.glob(file_pattern)
        filtered_files = filtered_files + [f for f in files if int(f.split('关联交易')[1][4:8]) > 630]

        related_party_transaction = []
        for file in files:
            try:
                # 尝试读取文件
                tmp_sht1 = pd.read_csv(file, encoding='gbk')
                related_party_transaction.append(tmp_sht1)
            except:
                related_party_transaction.append(pd.DataFrame())
        if len(related_party_transaction) > 0:
            tmp_sht = pd.concat(related_party_transaction, axis=0)
            tmp_sht = tmp_sht.reset_index(drop=True)
            tmp_sht = tmp_sht.groupby(by='S_INFO_WINDCODE')['S_RELATEDTRADE_AMOUNT'].sum().reset_index()
        else:
            tmp_sht = pd.DataFrame(columns=['S_INFO_WINDCODE', 'S_RELATEDTRADE_AMOUNT'])

        rpt_sht[y] = tmp_sht

    training_set = pd.DataFrame()
    for current_year in yearlist[1:]:
        print(current_year)
        features_this_year = annual_feature_engineering(current_year)
        training_set = pd.concat([training_set, features_this_year], axis=0)

    training_set = training_set.rename(columns={'S_INFO_WINDCODE': '证券代码'})
    features = pd.merge(cases, training_set, on=['证券代码', '年份'], how='left')

    features = features.drop_duplicates()
    features = features.dropna(axis=1, thresh=features.shape[0] // 2)
    # 所属行业转化为独热编码
    features = features.rename(columns={'所属行业': 'industries'})
    features = pd.get_dummies(features, columns=['industries'], drop_first=False)
    # 所有布尔数组转化为0/1
    for col in features.columns[4:]:
        if features[col].dtype == 'object' or 'bool':
            features[col] = features[col].astype('float32')

    features_predict = features.loc[features['年份'] == str(predict_year)].copy()
    features_predict = features_predict.reset_index(drop=True)
    features_predict.to_csv('./output/features_predict.csv')

    features_train = features.loc[features['年份'] != str(predict_year)].copy()
    features_train = features_train.reset_index(drop=True)
    features_train.to_csv('./output/features_train.csv')


# 描述性统计
# numeric_cols = features.select_dtypes(include=[np.number])
#
# description = numeric_cols.describe()
#
# # 计算更多的统计信息
# additional_stats = pd.DataFrame({
#     'Median': numeric_cols.median(),
#     'Mode': numeric_cols.mode().iloc[0],
#     'Variance': numeric_cols.var(),
#     'Skewness': numeric_cols.skew(),
#     'Kurtosis': numeric_cols.kurt(),
#     'nan_counts': numeric_cols.isnull().sum(),
#     '0 pct': (numeric_cols==0).sum()/numeric_cols.shape[0],
#     '1 pct': (numeric_cols==1).sum()/numeric_cols.shape[0]
# })
#
# additional_stats = additional_stats.T
#
# # 打印结果
# print("描述性统计：")
# print(description)
#
# print("\n更多统计信息：")
# print(additional_stats)
