import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
代码需要按顺序运行，但是只要前面运行过（已经生成文件），可以直接从中间步骤开始运行，方便调试。
其中，4.1)、4.2)、4.3) 三个模型训练过程不存在先后关系，互相独立。但是需要三个模型都训练完毕，才能运行5）
"""
# 1) 是否需要更新违规数据(是否更新了STK_Violation_Main.xlsx) --> 运行 dataset_obtain.py --> 生成 违规案例整理.xlsx
update_illegal = True
# 2) 违规数据描述性统计（仅用于课题报告撰写）--> 生成 fraud_data.xlsx
description_illegal = True
# 3) 是否需要更新财务数据(是否更新了dataset_path所在数据库) --> 运行 feature_engineering.py --> 生成 features_train.csv 和 features_predict.csv
update_finance = False
# 4.1) 是否需要更新logistic模型 --> 运行 logistic_train.py --> 生成 logit_model.pkl 和 features_importance_logistic.csv
update_logit = False
# 4.2) 是否需要更新mlp模型 --> 运行 MLP_train.py --> 生成 mlp.pkl 和 features_importance_mlp.csv
update_mlp = False
# 4.3) 是否需要更新xgboost模型 --> 运行 xgboost_train.py --> 生成 best_xgb_model.pkl 和 features_importance_xgboost.csv
update_xgb = False
# 5) 快捷输出三个模型的训练结果 --> 运行result_presenting.py --> 生成 features_importance_ave_rank.csv
show_result = False
# 6) 预测最新一年的违规概率 --> 运行 predict_model.py --> 生成 predict_result.csv
predict_new = False

dataset_path = r'D:/数据库'
start_year = '2001'      # 训练集案例最早的年份，需要保证导入的财务数据包含比参数更早1年的年份（需要计算同比）
end_year = '2022'        # 训练集案例最晚的年份，需要保证导入的财务数据包含参数值当年的年份
predict_year = '2023'    # word报告中的模型表现数据，不使用该年份；ppt中的模型预测中，预测该年份
split_test_size = '0.2'  # train_test_split(X, y, test_size=0.2, random_state=42)，保证多次分割的训练集相同
random_seed = '42'       # train_test_split(X, y, test_size=0.2, random_state=42)，保证多次分割的训练集相同
beta = '0.5'             # F_beta 范围(0,inf),在计算加权F1时，beta的取值；beta值越小，越关注精确率，反之，越关注召回率,当等于1时候同等。
                         # 2.5关注召回率; 0.5关注精确率
my_api = "sk-ad125357b57744468daefb32b23a8363"     # 使用阿里云通义千问的api。具体请见官方文档 https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

if __name__ == '__main__':

    '''   
    设置文件储存路径，回测开始结束日期（回测区间）
    split_test_size 和 random_seed 用于保证不同模型中，随机分割出来的训练集相同
    beta F_beta参数: (0,inf), 在计算加权F1时, beta值越小, 越关注准确率, 反之, 越关注召回率, 当等于1时同等重要
    '''

    assert start_year < end_year and predict_year > end_year, "Invalid year range"

    logging.info("Starting the main function")

    if update_illegal:
        logging.info("Updating illegal case data...")
        subprocess.run(['python', './function/dataset_obtain_csmar.py', 'STK_Violation_Main.xlsx', dataset_path])
        subprocess.run(['python', './function/dataset_obtain_wind_ai.py', dataset_path, my_api])
        subprocess.run(['python', './function/dataset_obtain_combine.py', dataset_path])

    if description_illegal:
        logging.info("Generating fraud data description...")
        subprocess.run(['python', './function/sample_description.py', start_year, end_year, dataset_path])

    if update_finance:
        logging.info("Updating financial data...")
        subprocess.run(['python', './function/feature_engineering.py', start_year, end_year, predict_year, dataset_path])

    if update_logit:
        logging.info("Training logistic regression model...")
        subprocess.run(['python', './function/logistic_train.py', split_test_size, random_seed, beta])

    if update_mlp:
        logging.info("Training MLP model...")
        subprocess.run(['python', './function/MLP_train.py', split_test_size, random_seed, beta])

    if update_xgb:
        logging.info("Training XGBoost model...")
        subprocess.run(['python', './function/xgboost_train.py', split_test_size, random_seed, beta])

    if show_result:
        logging.info("Presenting results...")
        subprocess.run(['python', './function/result_presenting.py', split_test_size, random_seed, beta])

    if predict_new:
        logging.info("Predicting results...")
        subprocess.run(['python', './function/result_predict.py', split_test_size, random_seed, beta])
