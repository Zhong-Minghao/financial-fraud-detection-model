import pandas as pd
import numpy as np
from training_toolkit import evaluate_result, significant_test, weighted_f1_score, get_sample_stats, get_best_threshold
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_tree
import joblib
import sys

df = pd.read_csv('./output/features_train.csv', index_col=0)
chinese_feature_name = pd.read_excel('./input/手动指标名称对照.xlsx')

# para_setting = pd.read_excel('./input/参数设置.xlsx', header=0)
# this_test_size = para_setting.loc[para_setting['名称'] == '训练集分割比例', '参数取值'].values[0]
# this_random = para_setting.loc[para_setting['名称'] == 'random_seed', '参数取值'].values[0]
# this_beta = para_setting.loc[para_setting['名称'] == 'beta', '参数取值'].values[0]
this_test_size = float(sys.argv[1])
this_random = int(sys.argv[2])
this_beta = float(sys.argv[3])

df_cleaned = df.copy()
numeric_cols = [col for col in df_cleaned if col not in ('证券代码', '年份', '证券简称', 'label')]
significant_features = significant_test(df_cleaned, numeric_cols)

# =============统计性描述=================
# data_stats = get_sample_stats(df_cleaned, numeric_cols)
# data_stats = data_stats.reset_index(drop=False)
# data_stats = data_stats.rename(columns={'index': '指标名称'})
# significant_result = significant_test(df_cleaned, numeric_cols, full_data=True)
# description = pd.merge(data_stats, significant_result, on='指标名称', how='outer')
# description = pd.merge(description, chinese_feature_name, on='指标名称', how='outer')
# description.to_csv('./output/description.csv')


# ========================logit========================
print('================logit=====================')
logit_model = joblib.load('./output/logit_model.pkl')
# 使用加载的模型进行预测
# 在进行预测前需要对新数据进行同样的预处理
X = df_cleaned[significant_features]
# X = df_cleaned[df_cleaned.columns[1:]]
X = X.fillna(0)
y = df_cleaned['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)

y_pred_prob_logit = logit_model.predict_proba(X_test)[:, 1]

best_threshold_logit = get_best_threshold(y_test, y_pred_prob_logit, this_beta)
y_pred_logit = np.where(y_pred_prob_logit >= best_threshold_logit, 1, 0)

# 计算准确率
print('训练集：')
y_pred_prob_train_logit = logit_model.predict_proba(X_train)[:, 1]
y_pred_train_logit = np.where(y_pred_prob_train_logit >= best_threshold_logit, 1, 0)
evaluate = evaluate_result(y_train, y_pred_train_logit, this_beta)
print('预测集：')
evaluate_logit = evaluate_result(y_test, y_pred_logit, this_beta)

# # 获取Logistic回归模型的系数
# coefficients = logit_model.coef_[0]
# # 将Logistic回归模型各因子的系数输出为DataFrame
# logit_importance_df = pd.DataFrame(coefficients,columns=['Coefficient'])
# logit_importance_df = logit_importance_df.reset_index()
# # 按系数绝对值排序，显示最重要的特征
# logit_importance_df['Absolute Coefficient'] = logit_importance_df['Coefficient'].abs()
# logit_importance_df = logit_importance_df.sort_values(by='Absolute Coefficient', ascending=False)
# print(logit_importance_df)

# ========================================================xgboost=======================================================
print('================xgboost=====================')
xgb_model = joblib.load('./output/best_xgb_model.pkl')
X = df_cleaned[significant_features]
y = df_cleaned['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)

dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

y_pred_prob_xgb = xgb_model.predict(dtest)
best_threshold_xgb = get_best_threshold(y_test, y_pred_prob_xgb, this_beta)

# 计算评估指标
print('训练集：')
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
y_pred_prob_train_xgb = xgb_model.predict(dtrain)
y_pred_train_xgb = (y_pred_prob_train_xgb > best_threshold_xgb).astype(int)
evaluate = evaluate_result(y_train, y_pred_train_xgb)
print('测试集：')
y_pred_xgb = (y_pred_prob_xgb > best_threshold_xgb).astype(int)
evaluate_xgb = evaluate_result(y_test, y_pred_xgb, this_beta)

# importance = xgb_model.get_score(importance_type='gain')
# importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# importance_df = pd.DataFrame(importance, columns=['指标名称', 'gain'])
# importance_df['指标名称'] = importance_df['指标名称'].str.replace('_y$', '', regex=True)
# importance_df = pd.merge(importance_df, chinese_feature_name, on='指标名称', how='outer')
# importance_df.loc[importance_df['中文含义'].isna(),'中文含义'] = importance_df.loc[importance_df['中文含义'].isna(),'指标名称']
# importance_df = importance_df.rename(columns={'gain': 'Importance'})
# importance_df.to_csv('./output/features_importance_xgboost.csv', index=False)

# 画出树,输出为pdf
# if False:
#     plt.figure(figsize=(20, 10))
#     plot_tree(xgb_model, num_trees=0)
#     plt.savefig('./output/xgboost_tree.pdf', format='pdf', dpi=2400)
#     plt.close()

# # AUC
# fpr, tpr, thresholds = roc_curve(y_test, preds)
#
# # 计算AUC
# roc_auc = auc(fpr, tpr)
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.show()



# ==============================================================MLP=====================================================
import torch
from torch.utils.data import DataLoader, TensorDataset
from MLP_train import MLP, get_mlp_feature_importance

X = df_cleaned[significant_features]
# 用MLP要去空值
X = X.fillna(0).values
y = df_cleaned['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1]
mlp_model = MLP(input_size).to(device)
mlp_model.load_state_dict(torch.load('./output/mlp.pth'))
print('===================MLP===================')
mlp_model.eval()
with torch.no_grad():
    this_outputs = mlp_model(X_test_tensor).to('cpu')
    y_pred_prob_mlp = torch.sigmoid(this_outputs).squeeze()
best_threshold_mlp = get_best_threshold(y_test, y_pred_prob_mlp, this_beta)

print('训练集：')
with torch.no_grad():
    this_outputs = mlp_model(X_train_tensor).to('cpu')
    y_pred_prob_train_mlp = torch.sigmoid(this_outputs).squeeze()
y_pred_train_mlp = np.where(y_pred_prob_train_mlp >= best_threshold_mlp, 1, 0)
evaluate = evaluate_result(y_train, y_pred_train_mlp, this_beta)

print('测试集：')
y_pred_mlp = np.where(y_pred_prob_mlp >= best_threshold_mlp, 1, 0)
evaluate_mlp = evaluate_result(y_test, y_pred_mlp, this_beta)

feature_names = significant_features  # 之前选择的重要特征

# 计算特征重要性
mlp_importance_df = get_mlp_feature_importance(mlp_model, feature_names)
mlp_importance_df = mlp_importance_df.rename(columns={'Feature': '指标名称'})
mlp_importance_df = pd.merge(mlp_importance_df, chinese_feature_name, on='指标名称', how='outer')

mlp_importance_df.to_csv('./output/features_importance_mlp.csv', index=False)


# ==================================================集成学习=======================================================
print('==========Ensemble Learning===========')
# softz
y_pred_prob_train_ensemble_df = pd.DataFrame({'logistic': y_pred_prob_train_logit, 'XGBoost': y_pred_prob_train_xgb, 'MLP': y_pred_prob_train_mlp})
y_pred_prob_ensemble_df = pd.DataFrame({'logistic': y_pred_prob_logit, 'XGBoost': y_pred_prob_xgb, 'MLP': y_pred_prob_mlp})

ensemble_weights = [evaluate_logit[-1], evaluate_xgb[-1], evaluate_mlp[-1]]
ensemble_weights = np.array(ensemble_weights) / np.sum(ensemble_weights)

y_pred_prob_train_ensemble = np.average(y_pred_prob_train_ensemble_df, axis=1, weights=ensemble_weights)
y_pred_prob_ensemble = np.average(y_pred_prob_ensemble_df, axis=1, weights=ensemble_weights)

best_threshold_ensemble = get_best_threshold(y_test, y_pred_prob_ensemble, this_beta)

print('训练集')
y_pred_train_ensemble = np.where(y_pred_prob_train_ensemble >= best_threshold_ensemble, 1, 0)
evaluate = evaluate_result(y_train, y_pred_train_ensemble, this_beta)
print('测试集')
y_pred_ensemble = np.where(y_pred_prob_ensemble >= best_threshold_ensemble, 1, 0)
evaluate_ensemble = evaluate_result(y_test, y_pred_ensemble, this_beta)

# hard
# y_pred_train_ensemble_df = pd.DataFrame({'logistic': y_pred_train_logit, 'XGBoost': y_pred_train_xgb, 'MLP': y_pred_train_mlp})
# y_pred_ensemble_df = pd.DataFrame({'logistic': y_pred_logit, 'XGBoost': y_pred_xgb, 'MLP': y_pred_mlp})
#
# y_pred_train_ensemble_hard = (y_pred_train_ensemble_df.sum(axis=1) > 2).astype(int)
# y_pred_ensemble_hard = (y_pred_ensemble_df.sum(axis=1) > 2).astype(int)
#
# print('训练集')
# evaluate = evaluate_result(y_train, y_pred_train_ensemble_hard, this_beta)
# print('测试集')
# evaluate_ensemble = evaluate_result(y_test, y_pred_ensemble_hard, this_beta)

# ============================输出特征重要性平均排名==========================================
impt_logit = pd.read_csv('./output/features_importance_logistic.csv')
impt_xgb = pd.read_csv('./output/features_importance_xgboost.csv')
impt_mlp = pd.read_csv('./output/features_importance_mlp.csv')

impt_logit = impt_logit.sort_values(by='Importance', ascending=False)
impt_xgb = impt_xgb.sort_values(by='Importance', ascending=False)
impt_mlp = impt_mlp.sort_values(by='Importance', ascending=False)

impt_logit['Rank'] = range(1, len(impt_logit) + 1)
impt_xgb['Rank'] = range(1, len(impt_xgb) + 1)
impt_mlp['Rank'] = range(1, len(impt_mlp) + 1)

assert '指标名称' in impt_logit.columns and '指标名称' in impt_xgb.columns and '指标名称' in impt_mlp.columns

# 合并数据
merged_df = pd.merge(impt_logit[['指标名称', 'Rank']],
                     impt_xgb[['指标名称', 'Rank']],
                     on='指标名称',
                     suffixes=('_logit', '_xgb'))
merged_df = pd.merge(merged_df,
                     impt_mlp[['指标名称', 'Rank']],
                     on='指标名称')
merged_df = merged_df.rename(columns={'Rank': 'Rank_mlp'})

# 计算平均序号
merged_df['Average_Rank'] = merged_df[['Rank_logit', 'Rank_xgb', 'Rank_mlp']].mean(axis=1)

merged_df = merged_df.sort_values(by='Average_Rank')

merged_df = pd.merge(merged_df, chinese_feature_name, on='指标名称')
merged_df = merged_df.drop_duplicates()

# 查看结果
print(merged_df[['指标名称', 'Average_Rank']].head())

merged_df.to_csv('./output/features_importance_ave_rank.csv')



