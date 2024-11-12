import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings("ignore")
#绘图需要
plt.rcParams['font.sans-serif'] = ['SimHei']

def get_sample_stats(this_df, cols_to_analyse):
    '''
    给出分标签的统计数据（均值和中位数）
    :param this_df: 包含'label'0/1标签列的待分析dataframe
    :param cols_to_analyse: 需要分析的列（包含'label'）
    :return:dataframe
    '''
    vio_sample = this_df.loc[this_df['label'] == 1, cols_to_analyse]
    control_sample = this_df.loc[this_df['label'] == 0, cols_to_analyse]
    stats = pd.DataFrame({
        'vio_sample mean': vio_sample.mean(),
        'vio_sample medium': vio_sample.median(),
        'control_sample mean': control_sample.mean(),
        'control_sample medium': control_sample.median()}
    )
    return stats


def data_preprocess(this_df):
    # 删除 NaN 值比例超过 10% 的列
    nan_ratio = this_df.isnull().mean()
    columns_to_drop = nan_ratio[nan_ratio > 0.10].index
    df_cleaned = this_df.drop(columns=columns_to_drop)
    return df_cleaned


def significant_test(data, features_cols, threshold=0.1, full_data=False):
    '''

    :param data:
    :param features_cols: 指标列
    :param threshold:
    :param full_data:
    :return:
    '''
    data = data.fillna(0)
    p_values = {}
    t_test = {}
    for column in features_cols:  # 排除代码，名称，label,行业列
        control_group = data[data['label'] == 0][column]
        fraud_group = data[data['label'] == 1][column]
        t_stat, p_value = stats.ttest_ind(control_group, fraud_group)
        p_values[column] = p_value
        t_test[column] = t_stat

    p_values_df = pd.DataFrame(list(p_values.items()), columns=['指标名称', 'P-Value'])
    t_test_df = pd.DataFrame(list(t_test.items()), columns=['指标名称', 'T-Test'])
    if full_data:
        result = pd.merge(p_values_df, t_test_df, on=['指标名称'])
        # result = result.loc[result['P-Value'] < threshold].copy()
        result = result.reset_index(drop=True)
        return result
    else:
        significant_features = p_values_df[p_values_df['P-Value'] < threshold]['指标名称'].tolist()
        print(f"通过显著性检验的特征（P值<{threshold}）：")
        print(significant_features)
        return significant_features


def weighted_f1_score(y_true, y_pred, beta=0.5):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision==0 or recall==0:
        return 0
    f1 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f1


def evaluate_result(testdata, predict, beta=0.5, show=True):
    # 计算评估指标
    accuracy = accuracy_score(testdata, predict)
    precision = precision_score(testdata, predict)
    recall = recall_score(testdata, predict)
    f1 = weighted_f1_score(testdata, predict, beta=beta)
    tn, fp, fn, tp = confusion_matrix(testdata, predict).ravel()

    if show:
        print(f"True Positives (TP): {tp}", end=' ')
        print(f"True Negatives (TN): {tn}", end=' ')
        print(f"False Positives (FP): {fp}", end=' ')
        print(f"False Negatives (FN): {fn}")
        print(f"准确率accuracy: {accuracy*100:.2f}%")
        print(f"精确率precision: {precision*100:.2f}%")
        print(f"召回率recall: {recall*100:.2f}%")
        print(f"特异性specificity: {tn / (tn + fp)*100:.2f}%")
        print(f"weighted f1 score: {f1:.4f}")

    return accuracy, precision, recall, f1


def multi_linear_test(data):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif["features"] = data.columns

    non_linear_features = vif.loc[vif['VIF Factor'] < 10, 'features'].tolist()
    return non_linear_features


def get_best_threshold(y_true, y_pred, this_beta=0.5):
    threshold_lower_bound = np.percentile(y_pred, 0.01)
    threshold_upper_bound = np.percentile(y_pred, 99.9)
    thresholds_range = np.linspace(threshold_lower_bound, threshold_upper_bound, 200)
    y_pred_ma = np.expand_dims(y_pred, axis=1)  # 扩展 y_pred 的维度以便与多个阈值进行比较
    y_pred_matrix = y_pred_ma >= thresholds_range
    scores = np.array([weighted_f1_score(y_true, y_pred_matrix[:, i], this_beta) for i in range(len(thresholds_range))])
    best_idx = np.argmax(scores)
    best_threshold = thresholds_range[best_idx]
    best_score = scores[best_idx]
    print(f"最佳f1得分：{best_score}, 最佳阈值: {best_threshold}")
    return best_threshold

if __name__ == '__main__':
    print('a')
