import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from training_toolkit import *
import joblib
import sys

# para_setting = pd.read_excel('./input/参数设置.xlsx', header=0)
# this_test_size = para_setting.loc[para_setting['名称'] == '训练集分割比例', '参数取值'].values[0]
# this_random = para_setting.loc[para_setting['名称'] == 'random_seed', '参数取值'].values[0]
# this_beta = para_setting.loc[para_setting['名称'] == 'beta', '参数取值'].values[0]

this_test_size = float(sys.argv[1])
this_random = int(sys.argv[2])
this_beta = float(sys.argv[3])

chinese_feature_name = pd.read_excel('./input/手动指标名称对照.xlsx')

df = pd.read_csv('./output/features_train.csv', index_col=0)
# =================特殊处理=================
# df_cleaned = data_preprocess(df)
df_cleaned = df.copy()

# 获取指标列
numeric_cols = [col for col in df_cleaned if col not in ('证券代码', '年份', '证券简称', 'label')]

# =============统计数据==================
# 统计数据
stats = get_sample_stats(df_cleaned, numeric_cols)

# 显著性检验（t检验）
significant_features = significant_test(df_cleaned, numeric_cols)
# significant_features = [col for col in significant_features if not col.startswith('industries')]

X = df_cleaned[significant_features]

X = X.fillna(0)

y = df_cleaned['label']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)

# ========================GAN===============================

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

lr = 0.0001
# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 开始训练
epochs = 10000
batch_size = 16

real_label = torch.ones(batch_size, 1).to(device)
smooth_factor = 0.1
real_label = real_label * (1 - smooth_factor)

fake_label = torch.zeros(batch_size, 1).to(device)

X_minority = df_cleaned.loc[df_cleaned['label']==1, significant_features].copy()
X_minority = X_minority.sample(frac=1).reset_index(drop=True)
X_minority = X_minority.fillna(0)

patience = 5000  # 允许的无改善轮数
counter = 0  # 计数器
best_g_loss = float('inf')
for epoch in range(epochs):
    # 训练判别器
    for _ in range(2):  # 平衡训练步骤
        idx = np.random.randint(0, X_minority.shape[0], batch_size)
        real_samples = torch.tensor(X_minority.iloc[idx].to_numpy(), dtype=torch.float32).to(device)

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(noise)

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_samples), real_label)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # 训练生成器
    optimizer_G.zero_grad()
    g_loss = criterion(discriminator(fake_samples), real_label)
    g_loss.backward()
    optimizer_G.step()

    # if g_loss.item() < best_g_loss:
    #     best_g_loss = g_loss.item()
    #     counter = 0  # 重置计数器
    #     torch.save(generator.state_dict(), './output/generator_checkpoint.pth')
    #     # torch.save(discriminator.state_dict(), './outout/discriminator_checkpoint.pth')
    # else:
    #     counter += 1

        # 检查是否达到早停条件
    # if counter >= patience:
    #     print(f"Early stopping at epoch {epoch}")
    #     break

    # 输出训练信息
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")


# 生成新的少数类样本
num_samples_to_generate = 50
# desired_ratio = 1 / 4
# num_samples_to_generate = int(X.shape[0] * desired_ratio - X_minority.shape[0])

noise = torch.randn(num_samples_to_generate, latent_dim).to(device)
generator = Generator(latent_dim, input_dim).to(device)

generated_samples = generator(noise).detach().cpu().numpy()

# 将生成的样本添加到训练集
X_train_resampled = pd.DataFrame(np.vstack([X_train, generated_samples]), columns=X_train.columns)
y_train_resampled = np.hstack([y_train, np.ones(num_samples_to_generate)])  # 标记为少数类

# ====================Logistics==================

print('===========logistic=============')
# 使用 statsmodels 进行 Logistic 回归
logit_model = LogisticRegression(max_iter=50)
logit_model.fit(X_train_resampled, y_train_resampled)

joblib.dump(logit_model, './output/logit_model.pkl')

# print(result.summary())

y_pred_prob = logit_model.predict_proba(X_test)[:,1]

# 确定最佳阈值
best_threshold = get_best_threshold(y_test, y_pred_prob, this_beta)
# 使用最佳阈值进行预测
y_pred_best = np.where(y_pred_prob >= best_threshold, 1, 0)
# 计算准确率
evaluate = evaluate_result(y_test, y_pred_best, this_beta)

# features importance
coefficients = logit_model.coef_[0]

# loadings = pca.components_
# features_importance = pd.DataFrame(coefficients.T @ loadings, columns=['Importance'])
# features_importance['指标名称'] = X.columns
# features_importance['Importance'] = features_importance['Importance'].abs()

features_importance = pd.DataFrame({
    '指标名称': X.columns,
    'Importance': abs(coefficients)
})
features_importance = features_importance.sort_values(by='Importance', ascending=False)
features_importance = pd.merge(features_importance, chinese_feature_name, on='指标名称', how='left')

features_importance.to_csv('./output/features_importance_logistic.csv', index=False)