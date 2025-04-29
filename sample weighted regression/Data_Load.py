import re
import pandas as pd
import numpy as np


def data_load(name):
    directory_path = "../dataset/"
    # 加载 混凝土数据集
    if name == "concrete":
        file_path = f'{directory_path}concrete_com_str/Concrete_Data.xls'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-1].values  # 取前八列作为特征
        y = df.iloc[:, -1].values  # 取最后一列作为标签

    # 加载 能源数据集1
    if name == "energy1":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values  # 取特征
        y = df.iloc[:, -2].values  # 取第一个回归值

    # 加载 能源数据集2
    if name == "energy2":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values  # 取特征
        y = df.iloc[:, -1].values  # 取第一个回归值

    # 加载 yacht数据集
    if name == "yacht":
        file_path = f'{directory_path}yacht_hydrodynamics/yacht_hydrodynamics.data'  # 请替换为你的文件路径
        df = pd.read_csv(file_path, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]  # 取特征
        y = data_array[:, -1]  # 取回归值

    # 加载 abalone鲍鱼数据集
    if name == "abalone":
        # # 打开 .names 文件查看数据集描述
        # with open("/data1/BCChen/abalone/abalone.names", 'r') as f:
        #     print(f.read())
        file_path = f'{directory_path}abalone/abalone.data'  # 请替换为你的文件路径
        df = pd.read_csv(file_path, header=None)
        # 使用 get_dummies 对 'Gender' 列进行独热编码
        df_encoded = pd.get_dummies(df, columns=[0], drop_first=False)  # drop_first=True 可以避免多重共线性
        y = df_encoded[4].values
        X = df_encoded.drop(columns=[4]).values

    # 加载 cps数据集  statlib
    if name == "cps":
        # #处理数据
        # file_path = "/data1/BCChen/cps/cps.txt"  # 请替换为你的文件路径
        # column_names = ['EDUCATION', 'SOUTH', 'SEX', 'EXPERIENCE', 'UNION', 'WAGE', 'AGE', 'RACE',
        #                 'OCCUPATION', 'SECTOR', 'MARR']
        # data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
        # # 使用 get_dummies 独热编码
        # data_encoded = pd.get_dummies(data, columns=['RACE', 'OCCUPATION', 'SECTOR'], drop_first=False)  # drop_first=True 可以避免多重共线性
        # data_encoded.to_excel("/data1/BCChen/cps/cps_encoded.xlsx", index=True)  # 将 DataFrame 写入 Excel 文件
        file_path =f'{directory_path}cps/cps_encoded.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        # 确保 'WAGE' 列和其他所有列的数据类型都是 float
        df = df.astype(float)
        y = df["WAGE"].values
        X = df.drop(columns=["WAGE"]).values
        y = y.astype(float)
        X = X.astype(float)

        # 加载 WINE-white数据集
    if name == "wine_white":
        file_path = f'{directory_path}wine_quality/winequality-white.csv'  # 请替换为你的文件路径
        data = pd.read_csv(file_path, sep=';', header=0)
        # print(data)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

        # 加载 WINE-red数据集
    if name == "wine_red":
        file_path = f'{directory_path}wine_quality/winequality-red.csv'  # 请替换为你的文件路径
        data = pd.read_csv(file_path, sep=';', header=0)
        # print(data)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

    # 加载 concrete-cs-slump数据集
    if name == "concrete-cs-slump":
        # 打开 .names 文件查看数据集描述
        # with open("/data1/BCChen/concrete_slump_test/slump_test.names", 'r') as f:
        #     print(f.read())
        # file_path = "/data1/BCChen/concrete_slump_test/slump_test.data"  # 请替换为你的文件路径
        # df = pd.read_csv(file_path, header=0, index_col=0)
        # df.to_excel("/data1/BCChen/concrete_slump_test/slump_test_done.xlsx", index=True)  # 将 DataFrame 写入 Excel 文件
        # print(df)
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[0]].values
        X = df.drop(columns=output_column).values

    # 加载 concrete-cs-flow数据集
    if name == "concrete-cs-flow":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[1]].values
        X = df.drop(columns=output_column).values

    # 加载 concrete-cs-mpa数据集
    if name == "concrete-cs-mpa":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[2]].values
        X = df.drop(columns=output_column).values

    # # 加载 IEMOCAP-V数据集
    if name == "IEMOCAP-V":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 0].values  # 取特征

    # # 加载 IEMOCAP-A数据集
    if name == "IEMOCAP-A":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 1].values  # 取特征

    # # 加载 IEMOCAP-D数据集
    if name == "IEMOCAP-D":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 2].values  # 取特征

    # # 加载 airfoil数据集
    if name == "airfoil":
        file_path_data =f'{directory_path}airfoil/airfoil_self_noise.dat'
        df = pd.read_csv(file_path_data, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]  # 取特征
        y = data_array[:, -1]  # 取回归值

    # # 加载 autopg数据集
    if name == "autompg":
        file_path = f'{directory_path}auto_mpg/auto-mpg.data'
        df = pd.read_csv(file_path, header=None, sep='\s+', na_values='?')

        # 仅对数值列填充缺失值
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # 处理非数值列（如汽车名称）
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna("unknown")

        # 提取特征和目标变量
        X = df.iloc[:, 1:-1].values  # 特征（从第2列开始，不要后面car name）
        y = df.iloc[:, 0].values  # 目标变量（第1列）


    # # 加载 housing数据集
    if name == "housing":
        file_path = f'{directory_path}housing/housing.xlsx'
        df = pd.read_excel(file_path, header=None)
        data = df.values  # 取特征

        # 处理数据
        processed_data = []
        for row in data:
            # 拆分字符串
            values = row[0].split()
            # 转换为浮点数
            float_values = [float(val) for val in values]
            processed_data.append(float_values)

        # 转换为NumPy数组
        data_array = np.array(processed_data)

        # 提取特征和目标变量
        X = data_array[:, :-1]  # 特征（除最后一列）
        y = data_array[:, -1]  # 目标变量（最后一列）


    print(X.shape)
    print(y.shape)
    # print(X.dtype)
    # print(y.dtype)
    # print(X)
    # print(y)
    # X = 0
    # y = 0
    return X, y


if __name__ == '__main__':
    data_load("housing")