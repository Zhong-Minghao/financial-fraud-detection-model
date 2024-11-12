import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from training_toolkit import significant_test, evaluate_result, get_best_threshold
import sys

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc4(x)  # 输出 logits
        return x


def get_mlp_feature_importance(model, feature_names):
    # 获取第一层的权重
    first_layer_weights = model.fc1.weight.cpu().detach().numpy()

    # 对每个特征取绝对值，并在不同的神经元之间取平均值
    feature_importance = np.mean(np.abs(first_layer_weights), axis=0)

    # 构建一个包含特征名和对应重要性的DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

# GAN 模型
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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 当验证损失不再改进时等待的 epoch 数。
            min_delta (float): 最小的损失改进量，如果小于此值则不视为改进。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == '__main__':

    df = pd.read_csv('./output/features_train.csv', index_col=0)
    # para_setting = pd.read_excel('./input/参数设置.xlsx', header=0)
    # this_test_size = para_setting.loc[para_setting['名称'] == '训练集分割比例', '参数取值'].values[0]
    # this_random = para_setting.loc[para_setting['名称'] == 'random_seed', '参数取值'].values[0]
    # this_beta = para_setting.loc[para_setting['名称'] == 'beta', '参数取值'].values[0]

    this_test_size = float(sys.argv[1])
    this_random = int(sys.argv[2])
    this_beta = float(sys.argv[3])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =================特殊处理=================
    # df_cleaned = data_preprocess(df)
    df_cleaned = df.copy()

    # 获取指标列
    numeric_cols = [col for col in df_cleaned if col not in ('证券代码', '年份', '证券简称', 'label')]
    # numeric_cols = [col for col in numeric_cols if not col.startswith('所属行业_')]

    significant_features = significant_test(df_cleaned, numeric_cols)
    X = df_cleaned[significant_features]
    # 用MLP要去空值
    X = X.fillna(0).values
    y = df_cleaned['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=this_test_size, random_state=this_random)

    # ================================GAN=======================================
    # 设置维度
    latent_dim = 100  # 噪声的维度
    input_dim = X_train.shape[1]  # 特征的维度

    # 初始化生成器和判别器
    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # 定义损失函数和优化器
    criterion_GAN = nn.BCELoss()
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
        real_loss = criterion_GAN(discriminator(real_samples), real_label)
        # 生成样本损失
        fake_loss = criterion_GAN(discriminator(fake_samples.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = criterion_GAN(discriminator(fake_samples), real_label)
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
        X_train_resampled = np.vstack([X_train, generated_samples])
        y_train_resampled = np.hstack([y_train, np.ones(num_samples_to_generate)])  # 标记为少数类

    else:
        X_train_resampled = X_train.copy()
        y_train_resampled = y_train.copy()

    # ============================================MLP===========================================
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    input_size = X_train.shape[1]
    model = MLP(input_size).to(device)

    # ===============criteria=====================
    # 1
    # criterion = nn.BCELoss()   # CrossEntropyLoss用于多分类，如果选择BCELoss，则在输出时应用sigmoid

    # 2
    # 为正类样本设置权重
    pos_weight = torch.tensor([3.0]).to(device)  # 为正类赋予3倍的权重
    # 使用 BCEWithLogitsLoss，给予正类样本更多权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 3
    # class FocalLoss(nn.Module):
    #     def __init__(self, alpha=1, gamma=2):
    #         super(FocalLoss, self).__init__()
    #         self.alpha = alpha
    #         self.gamma = gamma
    #
    #     def forward(self, inputs, targets):
    #         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    #         pt = torch.exp(-BCE_loss)
    #         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
    #         return torch.mean(F_loss)
    #
    # criterion = FocalLoss()

    # 为什么选择 Adam不用SGD
    # 快速收敛: 由于其自适应学习率机制，Adam 在复杂模型上往往能更快收敛，这对于缩短实验时间很有帮助。
    # 减少调参难度: Adam 对学习率不那么敏感，这可以减少手动调整学习率和其他超参数的工作。
    # 适应性强: 在处理异常检测这样的任务时，Adam 能够较好地应对数据的异质性和复杂性。
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # 训练模型
    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).to(device)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_val_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                val_outputs = model(inputs).to(device)
                val_loss = criterion(val_outputs.squeeze(), labels)

                total_val_loss += val_loss.item()  # 累加每个批次的验证损失
                num_batches += 1  # 统计批次数量

            average_val_loss = total_val_loss / num_batches if num_batches > 0 else 0  # 计算平均验证损失

        # 打印损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}")

        early_stopping(average_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('=======MLP========')
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).to('cpu').detach().numpy().reshape(-1)
    best_threshold = get_best_threshold(y_test, y_pred, this_beta)

    print('训练集：')
    with torch.no_grad():
        y_pred_train = model(X_train_tensor).to('cpu').detach().numpy().reshape(-1)
    y_pred_train = np.where(y_pred_train >= best_threshold, 1, 0)
    evaluate = evaluate_result(y_train_resampled, y_pred_train, this_beta)

    print('测试集：')
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).to('cpu').detach().numpy().reshape(-1)
    y_pred_test = np.where(y_pred_test >= best_threshold, 1, 0)
    evaluate = evaluate_result(y_test, y_pred_test, this_beta)

    # 保存模型
    PATH = './output/mlp.pth'
    torch.save(model.state_dict(), PATH)

    # 获取特征名称
    feature_names = significant_features  # 这是你之前选择的重要特征

    # 计算特征重要性
    feature_importance_df = get_mlp_feature_importance(model, feature_names)

    # # 输出特征重要性表格
    # print('feature_importance:')
    # print(feature_importance_df)
    chinese_feature_name = pd.read_excel('./input/手动指标名称对照.xlsx')
    feature_importance_df = feature_importance_df.rename(columns={'Feature': '指标名称'})
    feature_importance_df = pd.merge(feature_importance_df, chinese_feature_name, on='指标名称', how='left')
    # feature_importance_df.to_csv('./output/features_importance_mlp.csv', index=False)
