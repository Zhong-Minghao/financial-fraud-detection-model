import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from training_toolkit import evaluate_result, significant_test, get_best_threshold
import joblib

df = pd.read_csv('./output/features_train.csv', index_col=0)
df_predict = pd.read_csv('./output/features_predict.csv', index_col=0)
chinese_feature_name = pd.read_excel('./input/手动指标名称对照.xlsx')

# para_setting = pd.read_excel('./input/参数设置.xlsx', header=0)
# this_test_size = para_setting.loc[para_setting['名称'] == '训练集分割比例', '参数取值'].values[0]
# this_random = para_setting.loc[para_setting['名称'] == 'random_seed', '参数取值'].values[0]
# this_beta = para_setting.loc[para_setting['名称'] == 'beta', '参数取值'].values[0]
this_test_size = float(sys.argv[1])
this_random = int(sys.argv[2])
this_beta = float(sys.argv[3])

numeric_cols = [col for col in df if col not in ('证券代码', '年份', '证券简称', 'label')]
significant_features = significant_test(df, numeric_cols)

print('================logit=====================')
logit_model = joblib.load('./output/logit_model.pkl')
X = df_predict[significant_features]
X = X.fillna(0)
y = df_predict['label']

y_pred_prob_logit = logit_model.predict_proba(X)[:, 1]

best_threshold_logit = get_best_threshold(y, y_pred_prob_logit, this_beta)
y_pred_logit = np.where(y_pred_prob_logit >= best_threshold_logit, 1, 0)

evaluate_logit = evaluate_result(y, y_pred_logit, this_beta)

# ========================================================xgboost=======================================================
print('================xgboost=====================')
xgb_model = joblib.load('./output/best_xgb_model.pkl')
X = df_predict[significant_features]
y = df_predict['label']

dtest = xgb.DMatrix(X, label=y, missing=np.nan)

y_pred_prob_xgb = xgb_model.predict(dtest)
best_threshold_xgb = get_best_threshold(y, y_pred_prob_xgb, this_beta)

# 计算评估指标
y_pred_xgb = (y_pred_prob_xgb > best_threshold_xgb).astype(int)
evaluate_xgb = evaluate_result(y, y_pred_xgb, this_beta)

# ==============================================================MLP=====================================================
import torch
from torch.utils.data import DataLoader, TensorDataset
from MLP_train import MLP, get_mlp_feature_importance

X = df_predict[significant_features]
# 用MLP要去空值
X = X.fillna(0).values
y = df_predict['label'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = X.shape[1]
mlp_model = MLP(input_size).to(device)
mlp_model.load_state_dict(torch.load('./output/mlp.pth'))
print('===================MLP===================')
mlp_model.eval()
with torch.no_grad():
    this_outputs = mlp_model(X_test_tensor).to('cpu')
    y_pred_prob_mlp = torch.sigmoid(this_outputs).squeeze()
best_threshold_mlp = get_best_threshold(y, y_pred_prob_mlp, this_beta)

y_pred_mlp = np.where(y_pred_prob_mlp >= best_threshold_mlp, 1, 0)
evaluate_mlp = evaluate_result(y, y_pred_mlp, this_beta)

print('==========Ensemble Learning===========')
# soft
y_pred_prob_ensemble_df = pd.DataFrame({'logistic': y_pred_prob_logit, 'XGBoost': y_pred_prob_xgb, 'MLP': y_pred_prob_mlp})

ensemble_weights = [evaluate_logit[-1], evaluate_xgb[-1], evaluate_mlp[-1]]
ensemble_weights = np.array(ensemble_weights) / np.sum(ensemble_weights)

y_pred_prob_ensemble = np.average(y_pred_prob_ensemble_df, axis=1, weights=ensemble_weights)

best_threshold_ensemble = get_best_threshold(y, y_pred_prob_ensemble, this_beta)

y_pred_ensemble = np.where(y_pred_prob_ensemble >= best_threshold_ensemble, 1, 0)
evaluate_ensemble = evaluate_result(y, y_pred_ensemble, this_beta)

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

# ===========================================输出结果=================================================
result_df = df_predict[['证券代码', '年份', '证券简称']].copy()  # 保留股票代码
result_df['真实标签'] = y  # 添加真实标签
result_df['Logistic预测标签'] = y_pred_logit
result_df['MLP预测标签'] = y_pred_mlp
result_df['XGBoost预测标签（推荐）'] = y_pred_xgb
result_df['综合预测标签（推荐）'] = y_pred_ensemble
# 输出结果
print(result_df.head())

# 如果需要将结果保存为 CSV 文件
result_df.to_csv('./output/预测结果.csv', index=False)