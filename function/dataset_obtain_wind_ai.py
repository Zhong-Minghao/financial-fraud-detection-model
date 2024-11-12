import os
import glob
import sys
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import re
import pickle


def ai_obtain_violate_years(this_api, behavior):
    try:
        client = OpenAI(
            api_key=this_api,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system',
                 'content': '帮我提取下面这段文字中，财务报表造假事项发生的时间，以年份的格式返回，以分号间隔，如"2011;2021;2015"。注意事项1.对于文字描述为时间段（如2011年至2014年），则需要返回所有年份（如"2011;2012;2013;2014"）。注意事项2.若出现重复年份，只需要输出一次；若无发生时间相关文本，则返回N/A。注意事项3.不用输出为什么输出这个结果的原因。注意事项4.请仅返回财务报表造假年份，不是所有出现的时间都是造假年份。'},
                {'role': 'user', 'content': behavior}
            ],
        stream=True
        )
        full_content = ""
        for chunk in completion:
            # print(chunk.choices[0].delta.content)
            full_content += chunk.choices[0].delta.content
        return full_content

    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return 'Error'


def check_format(text):
    pattern = r"^(?:\d{4};)*\d{4}$"
    return bool(re.fullmatch(pattern, text))


def main_dataset_obtain(folder_path, this_api):

    # 读取最新的违规事件集合
    file_pattern = os.path.join(folder_path, f'wind中国A股重大事件/中国A股违规事件*.csv')
    files_list = glob.glob(file_pattern)

    files_list = sorted(
        files_list,
        key=lambda x: int(x[-12:-4])
    )

    try:
        with open('./input/last_wind_illegal_date.pkl', 'rb') as f:
            last_wind_illegal_date = pickle.load(f)
            last_wind_illegal_date = int(last_wind_illegal_date[-12:-4])
        print(f'上次运行代码时，更新至{last_wind_illegal_date}')

        last_wind_illegal_data = pd.read_csv('./output/违规案例整理_wind_ai.csv', index_col=0)
        print('==========导入历史记录：./output/wind_ai_违规案例整理.csv=============')

        filtered_files = [file for file in files_list if int(file[-12:-4]) > last_wind_illegal_date]

        if len(filtered_files)>0:
            with open('./input/last_wind_illegal_date.pkl', 'w') as f:
                pickle.dump(files_list[-1], f)

    except FileNotFoundError:
        print("不存在上次wind数据库读取记录，读取全部文件。")
        last_wind_illegal_data = pd.DataFrame()
        filtered_files = files_list


    illegal_data = []
    for file in filtered_files:
        try:
            # 尝试读取文件
            tmp_sht1 = pd.read_csv(file, encoding='gbk')
            illegal_data.append(tmp_sht1)
        except:
            illegal_data.append(pd.DataFrame())

    if len(illegal_data) == 0:
        print('没有需要处理的文件,已经为最新。')
        return last_wind_illegal_data
    else:
        tmp_sht = pd.concat(illegal_data, axis=0)

    illegal_df = tmp_sht.loc[tmp_sht['ILLEG_TYPE'].str.contains('虚假'), ['S_INFO_WINDCODE', 'BEHAVIOR']]
    illegal_df = illegal_df.drop_duplicates()
    illegal_df = illegal_df.reset_index(drop=True)

    this_year = []
    print('=================正在调用通义千问api====================')
    for index, row in tqdm(illegal_df.iterrows(), total=len(illegal_df), desc="进度"):
        this_year.append(ai_obtain_violate_years(this_api, row['BEHAVIOR']))
    illegal_df['年份'] = this_year

    # 输出ai识别文本与结果对照
    illegal_df.to_excel('./output/备注_ai年份判断的底稿.xlsx')

    # 仅筛选被成功识别的年份
    '''=============================在这一步将无法识别年份的案例都忽略掉了，日后改进方法中，可以在此加入人工筛选步骤======================='''
    illegal_df['年份'] = illegal_df['年份'].fillna('N/A')
    illegal_df['年份'] = illegal_df['年份'].apply(lambda x: x if check_format(x) else 'N/A')

    # 忽略没有年份为N/A的案例
    illegal_df = illegal_df.loc[illegal_df['年份'] != 'N/A']
    # 对得到的string结果进行排序 '2009;2010;2008' -> ['2008;2009;2010']
    illegal_df['年份'] = illegal_df['年份'].apply(lambda x: sorted(x.split(';'), key=int))

    # 数据透视
    fal_dlr_exploded = illegal_df[['S_INFO_WINDCODE', '年份']].copy()
    fal_dlr_exploded = fal_dlr_exploded.explode('年份')
    fal_dlr_exploded['年份'] = fal_dlr_exploded['年份'].astype(int)
    fal_dlr_exploded = fal_dlr_exploded.drop_duplicates()

    # 更新历史记录，保存为最新文件
    all_data = pd.concat([last_wind_illegal_data, fal_dlr_exploded], ignore_index=True)
    all_data = all_data.drop_duplicates()

    all_data.to_csv(r'./output/违规案例整理_wind_ai.csv')
    print('==========已导出./output/违规案例整理_wind_ai.csv=============')

    return fal_dlr_exploded


if __name__ == '__main__':
    folder_path = sys.argv[1]
    my_api = sys.argv[2]
    # folder_path = 'D:/数据库'
    # my_api = "sk-ad125357b57744468daefb32b23a8363"
    main_dataset_obtain(folder_path, my_api)


