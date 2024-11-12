import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from training_toolkit import significant_test, evaluate_result, get_best_threshold
from sklearn.metrics import roc_curve, auc
import optuna
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

this_boost_round = 1000
this_early_stopping_rounds = 10
# =================特殊处理=================
# df_cleaned = data_preprocess(df)
df_cleaned = df.copy()

# 获取指标列
numeric_cols = [col for col in df_cleaned if col not in ('证券代码', '年份', '证券简称', 'label')]
# numeric_cols = [col for col in numeric_cols if not col.startswith('所属行业_')]

significant_features = significant_test(df_cleaned, numeric_cols)
X = df_cleaned[significant_features]
# X = df_cleaned[numeric_cols]
# X = df_cleaned[[col for col in df_cleaned.columns if (col != '证券代码' and col != 'label' and col != '年份')]]

y = df_cleaned['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置维度
latent_dim = 100  # 噪声的维度
input_dim = X_train.shape[1]  # 特征的维度

# 初始化生成器和判别器
generator = Generator(latent_dim, input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# 开始训练
epochs = 10000
batch_size = 16

real_label = torch.ones(batch_size, 1).to(device)
fake_label = torch.zeros(batch_size, 1).to(device)

X_minority = df_cleaned.loc[df_cleaned['label'] == 1, significant_features].copy()
X_minority = X_minority.sample(frac=1).reset_index(drop=True)
X_minority = X_minority.fillna(0)

for epoch in range(epochs):
    # 训练判别器
    idx = np.random.randint(0, X_minority.shape[0], batch_size)
    real_samples = torch.tensor(X_minority.iloc[idx].to_numpy(), dtype=torch.float32).to(device)

    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = generator(noise)

    optimizer_D.zero_grad()
    # 真实样本损失
    real_loss = criterion(discriminator(real_samples), real_label)
    # 生成样本损失
    fake_loss = criterion(discriminator(fake_samples.detach()), fake_label)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # 训练生成器
    optimizer_G.zero_grad()
    g_loss = criterion(discriminator(fake_samples), real_label)
    g_loss.backward()
    optimizer_G.step()

    # 输出训练信息
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# 生成新的少数类样本
desired_ratio = 1 / 4
num_samples_to_generate = int(X.shape[0] * desired_ratio - X_minority.shape[0])

if num_samples_to_generate > 0:
    noise = torch.randn(num_samples_to_generate, latent_dim).to(device)
    generated_samples = generator(noise).detach().cpu().numpy()

    # 将生成的样本添加到训练集
    X_train_resampled = pd.DataFrame(np.vstack([X_train, generated_samples]), columns=X_train.columns)
    y_train_resampled = np.hstack([y_train, np.ones(num_samples_to_generate)])  # 标记为少数类

else:
    X_train_resampled = X_train.copy()
    y_train_resampled = y_train.copy()

dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled, missing=np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)


def objective(trial):
    # 设置需要调优的参数
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', sum(y_train == 0) / sum(y_train) * 0.25,
                                                sum(y_train == 0) / sum(y_train)),
        'lambda': trial.suggest_float('lambda', 0.5, 2),
        'alpha': trial.suggest_float('alpha', 0.5, 2)

    }

    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=this_boost_round,
        nfold=5,  # 5折交叉验证
        early_stopping_rounds=this_early_stopping_rounds,  # 早停
        metrics='auc',  # 评估标准
        as_pandas=True,
        verbose_eval=False
    )

    # 获取最后一轮的的AUC得分
    score = cv_results['test-auc-mean'].values[-1]

    return score


# 创建Optuna研究对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 输出最佳参数
best_params = study.best_params
print(f"最佳参数: {best_params}")
best_params = {**best_params, 'objective': 'binary:logistic',
               'eval_metric': 'auc',
               'booster': 'gbtree',
               'tree_method': 'hist',
               'device': 'cuda'}

print('===========xgboost + gbtree + para-tuning=============')
kf = KFold(n_splits=5, shuffle=True)

cv_results = xgb.cv(
    best_params,
    dtrain,
    num_boost_round=this_boost_round,
    folds=kf,
    early_stopping_rounds=this_early_stopping_rounds,
    verbose_eval=False,
    as_pandas=True
)

model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=len(cv_results),
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=this_early_stopping_rounds,
    verbose_eval=False
)

y_pred = model.predict(dtest)
best_threshold = get_best_threshold(y_test, y_pred, this_beta)

print('训练集：')
y_preds_train = model.predict(dtrain)
y_preds_binary_train = (y_preds_train > best_threshold).astype(int)
evaluate = evaluate_result(y_train_resampled, y_preds_binary_train, this_beta)
print('测试集：')
y_pred_binary = (y_pred > best_threshold).astype(int)
evaluate = evaluate_result(y_test, y_pred_binary, this_beta)

# ===================================================

plt.rcParams['font.sans-serif'] = ['SimHei']

# 保存模型到文件
joblib.dump(model, './output/best_xgb_model.pkl')

importance = model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

importance_df = pd.DataFrame(importance, columns=['指标名称', 'gain'])
importance_df['指标名称'] = importance_df['指标名称'].str.replace('_y$', '', regex=True)
importance_df = pd.merge(importance_df, chinese_feature_name, on='指标名称', how='outer')
importance_df.loc[importance_df['中文含义'].isna(),'中文含义'] = importance_df.loc[importance_df['中文含义'].isna(),'指标名称']
importance_df = importance_df.rename(columns={'gain': 'Importance'})
importance_df.to_csv('./output/features_importance_xgboost.csv', index=False)

# AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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
#
# # 计算 Youden's J statistic
# J = tpr - fpr
#
# # 找到最大 J 对应的阈值
# best_threshold_index = np.argmax(J)
# best_threshold = thresholds[best_threshold_index]
#
# print(f"最佳阈值: {best_threshold}")