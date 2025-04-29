# 加入baseline  ramdom

import warnings
import random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import resample
from KMEANS_GPU import KMeansGPU
from Data_Load import *
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter("ignore")
class sw_strategy:
    def __init__(self, maxN, minN, deviceGPU, type, random_state, nBoots=10):
        self.type = type # 选择样本方法
        self.device = deviceGPU # 使用的GPU
        self.minN = minN # 最小选择样本数量
        self.maxN = maxN # 最大选择样本数量
        self.budget = self.maxN - self.minN  # 选择样本数量
        self.nBoots = nBoots # 迭代次数
        self.random_state = random_state

    def fit(self, XPool, YPool):
        # 按照2:8的比例划分数据集
        X_train, X_test, y_train, y_test = train_test_split(XPool, YPool, test_size=0.2, random_state=self.random_state)
        self.scaler = StandardScaler()# 数据标准化：初始化标准化器
        X_train = self.scaler.fit_transform(X_train)# 使用训练数据拟合标准化器，并将训练数据进行标准化
        X_test = self.scaler.transform(X_test)# 使用相同的标准化器转换测试数据

        self.numPool = X_train.shape[0] # 样本数量
        self.weight_mat = np.full((self.numPool,), 1 / self.numPool)  # 初始化权重矩阵
        self.idsselect = np.zeros(self.maxN, dtype=int)  # 一维数组，存储训练样本的索引
        self.idsunselect = np.arange(self.numPool)  # 一维数组，存储未选择的样本索引

        distX = self.compute_dist(X_train)# 计算距离矩阵

        self.select_initial_samples(distX, X_train)# 选择初始样本
        self.select_next_sample(distX, X_train, y_train)# 选择后续样本
        rmse_array, cc_array = self.evaluation(X_train, y_train, X_test, y_test)

        return rmse_array, cc_array

    def compute_dist(self, X_pool):
        X_pool_torch = torch.tensor(X_pool, device=self.device)
        distX = torch.norm(X_pool_torch[:, np.newaxis] - X_pool_torch, dim=2)

        return distX.cpu().numpy()

    def weight_normalize(self):
        self.weight_mat = self.weight_mat / np.sum(self.weight_mat)

    def select_initial_samples(self, distX, X_train):
        #初始化流程
        if self.type in (0, 1, 2, 3):#GSX初始化
            # 选择初始样本
            dist = np.mean(distX, axis=1)  # 每个样本的平均距离
            idx = np.argmin(dist)
            self.idsselect[0] = self.idsunselect[idx]  # 选择距离最小的第一个样本
            self.idsunselect = np.delete(self.idsunselect, idx)  # 更新未选择样本集

            for i in range(1, self.minN):
                dist = np.min(distX[np.ix_(self.idsunselect, self.idsselect[0:i])], axis=1)  # 计算距离
                idx = np.argmax(dist)  # 选择最远的样本
                self.idsselect[i] = self.idsunselect[idx]  # 更新训练样本集
                self.idsunselect = np.delete(self.idsunselect, idx)  # 移除已选择的样本

        if self.type in (4, 5):#rd初始化
            n_clusters = X_train.shape[1]
            X_train_torch = torch.from_numpy(X_train).to(self.device)
            cluster_ids_x, cluster_centers = KMeansGPU(num_clusters=n_clusters, random_state=None, max_iter=300,
                                                       device=torch.device(self.device)).fit(X_train_torch)
            labels_pred = cluster_ids_x.cpu().numpy()
            cluster_labels, count = np.unique(labels_pred, return_counts=True)
            cluster_dict = OrderedDict()  # 定义有序字典

            for i in cluster_labels:  # 遍历簇，形成一个集合
                cluster_dict[i] = []  # 每个簇用预测结果 i 代表
            for i in range(len(labels_pred)):
                cluster_dict[labels_pred[i]].append(i)  # 将每个集合都放入

            for i in cluster_labels:
                cluster_samples_idx = torch.tensor(cluster_dict[i], device=self.device)  # 获取当前簇的样本索引
                cluster_samples = X_train_torch[cluster_samples_idx].float()  # 提取当前簇的样本
                centroid = cluster_samples.mean(dim=0)  # 计算簇中心（均值）
                distances = torch.norm(cluster_samples - centroid, dim=1)  # 计算每个样本到簇中心的欧几里得距离
                closest_idx = torch.argmin(distances)  # 找到距离簇中心最近的样本

                tar_idx = cluster_dict[i][closest_idx.item()]
                self.idsselect[i] = tar_idx  # 更新训练样本集
                self.idsunselect = np.delete(self.idsunselect, np.where(self.idsunselect == tar_idx))  # 移除已选择的样本

        if self.type == 6:  #随机选择
            for i in range(self.minN):
                tar_idx = random.choice(self.idsunselect)
                self.idsselect[i] = tar_idx  # 更新训练样本集
                self.idsunselect = np.delete(self.idsunselect, np.where(self.idsunselect == tar_idx))  # 移除已选择的样本

        self.weight_mat[self.idsselect[:self.minN]] = 0  # 将已标记样本的权重设置为 0
        self.weight_normalize()

    def select_next_sample(self, distX, X_train, y_train):
        """
        选择下一个样本
        :param n: 当前选择的样本数量
        :return: None
        """
        for n in range(self.minN, self.maxN):
            alpha_set = 1e-3

            # QBC
            if self.type == 0:
                C = np.maximum(1, np.ceil(n * np.random.rand(self.nBoots, n)).astype(int)) -1  # 随机选择样本
                QBC = np.zeros(len(self.idsunselect))  # 初始化QBC值
                Ys = np.zeros((len(self.idsunselect), self.nBoots))  # 用于存储每次bootstrap的预测结果

                for i in range(self.nBoots):
                    b = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[C[i, :]], :], y_train[self.idsselect[C[i, :]]])
                    b_coef_ = np.hstack((b.intercept_, b.coef_.flatten()))  # 偏置项和系数
                    Ys[:, i] = np.column_stack([np.ones(len(self.idsunselect)), X_train[self.idsunselect]]) @ b_coef_

                QBC = np.var(Ys, axis=1)

                # 选择方差最大（即最不确定）的样本，并将其添加到训练集
                idx = np.argmax(QBC)
                self.idsselect[n] = self.idsunselect[idx]  # 更新训练集
                self.idsunselect = np.delete(self.idsunselect, idx)  # 移除已选择的样本

            # sw-QBC
            if self.type == 1:
                C = np.maximum(1, np.ceil(n * np.random.rand(self.nBoots, n)).astype(int)) -1 # 随机选择样本
                QBC = np.zeros(len(self.idsunselect))  # 初始化QBC值
                Ys = np.zeros((len(self.idsunselect), self.nBoots))  # 用于存储每次bootstrap的预测结果

                for i in range(self.nBoots):
                    b = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[C[i, :]], :], y_train[self.idsselect[C[i, :]]])
                    b_coef_ = np.hstack((b.intercept_, b.coef_.flatten()))  # 偏置项和系数
                    Ys[:, i] = np.column_stack([np.ones(len(self.idsunselect)), X_train[self.idsunselect]]) @ b_coef_

                QBC = np.var(Ys, axis=1)
                for i in range(len(self.idsunselect)):
                    QBC[i] *= self.weight_mat[self.idsunselect[i]]

                # 选择方差最大（即最不确定）的样本，并将其添加到训练集
                idx = np.argmax(QBC)
                tar_idx = self.idsunselect[idx]

                # 聚类 找到当前选择样本所在簇 对该簇实行权重更新
                X_pool_torch = torch.tensor(X_train, device=self.device)  # 放入GPU
                cluster_ids_x, cluster_centers = KMeansGPU(num_clusters=n, random_state=None, max_iter=300,
                                                           device=torch.device(self.device)).fit(X_pool_torch)
                labels_pred = cluster_ids_x.cpu().numpy()
                # labels_pred = KMeans(n_clusters=n).fit_predict(self.X_pool)  # 聚类操作，类的大小为目前选择的样本数量
                tar_cluster_label = labels_pred[tar_idx]  # 选中样本所在簇标签
                tar_cluster_idx = np.where(labels_pred == tar_cluster_label)[0]  # 目标簇中的样本索引

                distances = distX[np.ix_(tar_cluster_idx, [tar_idx])].flatten()  # 计算距离
                # distances = np.linalg.norm(self.X_pool[tar_cluster_idx] - self.X_pool[tar_idx],axis=1)  # 计算每个样本到当前已标记样本的距离
                distances_norm = distances / np.max(distances)
                for i in range(len(X_train[tar_cluster_idx])):
                    self.weight_mat[tar_cluster_idx[i]] *= distances_norm[i]  # 根据距离计算权重，距离越近，权重越小
                # self.weight_mat[tar_idx] = 0
                self.weight_normalize()

                self.idsselect[n] = self.idsunselect[idx]  # 更新训练集
                self.idsunselect = np.delete(self.idsunselect, idx)  # 移除已选择的样本

            # emcm
            if self.type == 2:
                C = np.maximum(1, np.ceil(n * np.random.rand(self.nBoots, n)).astype(int)) - 1  # 随机选择样本
                EMCM = np.zeros(len(self.idsunselect))  # 初始化EMCM值
                Ys = np.zeros((len(self.idsunselect), self.nBoots))  # 用于存储每次bootstrap的预测结果

                b0 = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[:n], :], y_train[self.idsselect[:n]])
                Y0 = b0.predict(X_train[self.idsunselect])

                for i in range(self.nBoots):
                    b = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[C[i, :]], :], y_train[self.idsselect[C[i, :]]])
                    Ys[:, i] = b.predict(X_train[self.idsunselect])

                # 计算EMCM值
                for i in range(len(self.idsunselect)):
                    for j in range(self.nBoots):
                        EMCM[i] += np.linalg.norm((Ys[i, j] - Y0[i]) * X_train[self.idsunselect[i]])

                # 选择EMCM最大值的样本
                idx = np.argmax(EMCM)
                self.idsselect[n] = self.idsunselect[idx]  # 更新训练集
                self.idsunselect = np.delete(self.idsunselect, idx)  # 移除已选择的样本

            # sw-emcm
            if self.type == 3:
                C = np.maximum(1, np.ceil(n * np.random.rand(self.nBoots, n)).astype(int)) - 1  # 随机选择样本
                EMCM = np.zeros(len(self.idsunselect))  # 初始化EMCM值
                Ys = np.zeros((len(self.idsunselect), self.nBoots))  # 用于存储每次bootstrap的预测结果

                b0 = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[:n], :], y_train[self.idsselect[:n]])
                Y0 = b0.predict(X_train[self.idsunselect])

                for i in range(self.nBoots):
                    b = Ridge(alpha=alpha_set).fit(X_train[self.idsselect[C[i, :]], :], y_train[self.idsselect[C[i, :]]])
                    Ys[:, i] = b.predict(X_train[self.idsunselect])

                # 计算EMCM值
                for i in range(len(self.idsunselect)):
                    for j in range(self.nBoots):
                        EMCM[i] += np.linalg.norm((Ys[i, j] - Y0[i]) * X_train[self.idsunselect[i]])
                    EMCM[i] *= self.weight_mat[self.idsunselect[i]]

                # 选择EMCM最大值的样本
                idx = np.argmax(EMCM)
                tar_idx = self.idsunselect[idx]

                # 聚类 找到当前选择样本所在簇 对该簇实行权重更新
                X_pool_torch = torch.tensor(X_train, device=self.device)  # 放入GPU
                cluster_ids_x, cluster_centers = KMeansGPU(num_clusters=n, random_state=None, max_iter=300,
                                                           device=torch.device(self.device)).fit(X_pool_torch)
                labels_pred = cluster_ids_x.cpu().numpy()
                tar_cluster_label = labels_pred[tar_idx]  # 选中样本所在簇标签
                tar_cluster_idx = np.where(labels_pred == tar_cluster_label)[0]  # 目标簇中的样本索引

                distances = distX[np.ix_(tar_cluster_idx, [tar_idx])].flatten()  # 计算距离
                distances_norm = distances / np.max(distances)
                for i in range(len(X_train[tar_cluster_idx])):
                    self.weight_mat[tar_cluster_idx[i]] *= distances_norm[i]  # 根据距离计算权重，距离越近，权重越小
                self.weight_normalize()

                self.idsselect[n] = self.idsunselect[idx]  # 更新训练集
                self.idsunselect = np.delete(self.idsunselect, idx)  # 移除已选择的样本

            # rd
            if self.type == 4:
                n_clusters = n + 1

                X_pool_torch = torch.tensor(X_train, device=self.device)  # 放入GPU
                cluster_ids_x, cluster_centers = KMeansGPU(num_clusters=n_clusters, random_state=None, max_iter=300,device=torch.device(self.device)).fit(X_pool_torch)
                labels_pred = cluster_ids_x.cpu().numpy()
                cluster_labels, count = np.unique(labels_pred, return_counts=True)  # 获取每个簇的标签和样本数

                cluster_dict = OrderedDict()  # 定义有序字典
                for i in cluster_labels:  # 遍历簇，形成一个集合
                    cluster_dict[i] = []  # 每个簇用预测结果 i 代表
                for idx in self.idsselect[:n]:
                    cluster_dict[labels_pred[idx]].append(idx)  # 将已经标记好的样本添加进入

                empty_ids = OrderedDict()  # 定义一个空字典
                for i in cluster_labels:  # 遍历簇
                    if len(cluster_dict[i]) == 0:  # 判断此时簇是否为空，如果含有标记样本则为空，没有标记样本则不为空
                        idx = np.where(cluster_labels == i)[0][0]
                        empty_ids[i] = count[idx]  # 记录空簇的未标记样本的数量

                # 判断是否存在空簇
                if empty_ids:
                    tar_label = max(empty_ids, key=empty_ids.get)# 如果有空簇，选择样本数量最多的空簇
                else:
                    print("没有出现不包含标记样本的簇")

                tar_cluster_ids = []  # 将在 tar_label 簇中的样本加入到集合 tar_cluster_ids 中
                for idx in range(self.numPool):  # 遍历样本数量
                    if labels_pred[idx] == tar_label:  # 判断样本是否在 tar_label 中
                        tar_cluster_ids.append(idx)

                # 找到目标簇中的中心点
                tar_cluster_ids_torch = torch.tensor(tar_cluster_ids,device=self.device)  # 将tar_cluster_ids转换为Tensor并移到GPU
                tar_samples = X_pool_torch[tar_cluster_ids_torch].float()
                centroid = tar_samples.mean(dim=0)  # 计算簇中心
                distances = torch.norm(tar_samples - centroid, dim=1)  # 计算tar_cluster_ids中所有样本到簇中心的距离
                close_dist, closest_idx = torch.min(distances, dim=0)  # 找到距离最小的样本
                tar_idx = tar_cluster_ids[closest_idx.item()]  # 获取最近样本的索引

                self.idsselect[n] = tar_idx  # 更新训练集
                self.idsunselect = self.idsunselect[self.idsunselect != tar_idx]  # 移除已选择的样本

            # sw-rd
            if self.type == 5:
                n_clusters = n + 1

                X_pool_torch = torch.tensor(X_train, device=self.device)# 放入GPU
                weight_mat_torch = torch.from_numpy(self.weight_mat).to(self.device)  # 权重矩阵移到GPU
                cluster_ids_x, cluster_centers = KMeansGPU(num_clusters=n_clusters, random_state=None, max_iter=300,
                                                           device=torch.device(self.device)).fit(X_pool_torch)
                labels_pred = cluster_ids_x.cpu().numpy()
                cluster_labels, count = torch.unique(torch.tensor(labels_pred, device=self.device), return_counts=True)# 获取每个簇的标签和数量

                tar_cluster_label = None
                weight_sum_max = 0
                labels_pred_tensor = torch.tensor(labels_pred, device=self.device)
                for cluster_label in cluster_labels:
                    cluster_indices = torch.where(labels_pred_tensor == cluster_label)[0]
                    weight_sum = torch.sum(weight_mat_torch[cluster_indices])  # 计算总权重
                    if weight_sum > weight_sum_max and len(cluster_indices) > 1:
                        weight_sum_max = weight_sum
                        tar_cluster_label = cluster_label.item()  # 找到总权重最大的目标簇

                tar_cluster_idx = torch.where(labels_pred_tensor == tar_cluster_label)[0]#目标簇中的样本索引
                weight_max = torch.max(weight_mat_torch[tar_cluster_idx])#发现目标簇中的最大权重
                tar_idxs = tar_cluster_idx[torch.where(weight_mat_torch[tar_cluster_idx] == weight_max)[0]]#找到权重最大的样本索引

                if len(tar_idxs) == 1:
                    tar_idx = tar_idxs[0].item()
                else:
                    tar_samples = X_pool_torch[tar_idxs].float()
                    centroid = torch.mean(tar_samples, dim=0)  # 计算中心
                    distances = torch.norm(tar_samples - centroid, dim=1)  # 计算每个样本到簇中心的距离
                    closest_idx = torch.argmin(distances)  # 找到距离最近的样本
                    tar_idx = tar_idxs[closest_idx].item()

                # 权重更新，已标记样本簇的权重减小，新标记样本权重置为0,并归一化
                # 简单使用距离的归一化
                distances = torch.norm(X_pool_torch[tar_cluster_idx] - X_pool_torch[tar_idx], dim=1)  # 计算每个样本到已标记样本的距离
                if len(distances) > 1:
                    distances = distances / torch.max(distances)  # 归一化距离
                for i in range(len(tar_cluster_idx)):
                    weight_mat_torch[tar_cluster_idx[i]] *= distances[i]  # 根据距离计算权重，距离越近，权重越小
                self.weight_mat = weight_mat_torch.cpu().numpy()
                self.weight_normalize()

                # 将选出的样本进行标记
                self.idsselect[n] = tar_idx  # 更新训练集
                self.idsunselect = self.idsunselect[self.idsunselect != tar_idx]  # 移除已选择的样本

            # random
            if self.type == 6:
                tar_idx = random.choice(self.idsunselect)
                self.idsselect[n] = tar_idx  # 更新训练样本集
                self.idsunselect = np.delete(self.idsunselect, np.where(self.idsunselect == tar_idx))  # 移除已选择的样本


    def evaluation(self, X_pool, y_pool, X_test, y_test):
        """
        使用岭回归训练模型并评估其性能
        :return: 当前模型的RMSE cc
        """
        rmse_value_array = []
        cc_value_array = []

        for i in range(self.budget + 1):
            X_train = X_pool[self.idsselect[0:i + self.minN]]
            y_train = y_pool[self.idsselect[0:i + self.minN]]

            alpha_set = 1e-3
            model_ridge = Ridge(alpha=alpha_set)
            model_ridge.fit(X_train, y_train)  # 训练回归模型
            y_pred_test = model_ridge.predict(X_test)  # 测试集预测

            rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))# 计算RMSE
            cc = np.corrcoef(y_pred_test, y_test)[0, 1]# 计算cc

            rmse_value_array.append(rmse)
            cc_value_array.append(cc)  # (budget+1,)

        return rmse_value_array, cc_value_array

def train(X, y, experiment_name, maxN, minN, iteration, deviceGPU_set):
    budget_set = maxN - minN
    iteration = iteration
    type_num = len(experiment_name)
    rmse = np.zeros((type_num, budget_set + 1, iteration))
    cc = np.zeros((type_num, budget_set + 1, iteration))
    ramdom_array = []

    for i in range(iteration):
        random_state = random.randint(0, iteration*100) #随机种子
        ramdom_array.append(random_state)
        for j in range(type_num):
            model = sw_strategy(maxN=maxN, minN=minN, type=j, deviceGPU=deviceGPU_set, random_state=random_state)
            rmse_array, cc_array = model.fit(XPool=X, YPool=y)  # (budget+1,)
            for budget in range(budget_set + 1):
                rmse[j, budget, i] = rmse_array[budget]
                cc[j, budget, i] = cc_array[budget]
                if budget % 10 == 0: #打印
                    print(f"type ={experiment_name[j]} 迭代到第{i}次,budget为{budget}时,测试集rmse为{rmse[j, budget, i]}")
                    print(f"type ={experiment_name[j]} 迭代到第{i}次,budget为{budget}时,测试集cc为{cc[j, budget, i]}")


    rmse_mean = np.mean(rmse, axis=2)
    cc_mean = np.mean(cc, axis=2)
    rmse_std = np.std(rmse, axis=2)

    return rmse_mean, cc_mean, ramdom_array, rmse_std
def train_base(X, y, budget_set, iteration, random_array):
    # 用全部数据来训练一个上限
    rmse_base = []
    cc_base = []
    for i in range(iteration):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_array[i])
        model_base = Ridge(alpha=0.001)
        model_base.fit(X_train, y_train)
        y_pred_test = model_base.predict(X_test)  # 测试集预测
        rmse_base.append(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
        cc_base.append(np.corrcoef(y_pred_test, y_test)[0, 1])
    rmse_base_mean = np.mean(rmse_base)

    rmse_base_array = [rmse_base_mean] * (budget_set + 1)
    cc_base_mean = np.mean(cc_base)
    cc_base_array = [cc_base_mean] * (budget_set + 1)
    print(f"base_rmse为{rmse_base_mean}")
    print(f"base_cc为{cc_base_mean}")

    return rmse_base_array, cc_base_array
def plot_results(rmse_base_array, cc_base_array, rmse_mean, cc_mean, experiment_name, budget_set, type_num, dataname,time):
    # rmse_plt
    plt.plot(range(budget_set + 1), rmse_base_array, label=f'minimum')
    for i in range(type_num):
        plt.plot(range(len(rmse_mean[i])), rmse_mean[i], label=f'{experiment_name[i]}')
    plt.xlabel('m')  # 添加横轴标签
    plt.ylabel('RMSE')  # 添加纵轴标签
    plt.title(f"rmse-{dataname}-{time}")
    plt.legend()
    image_save_path = "../output/picture/"#本地保存路径
    plt.savefig(f"{image_save_path}rmse-{dataname}-{time}.png")
    plt.show()

    # cc_plt
    plt.plot(range(budget_set + 1), cc_base_array, label=f'maximum')
    for i in range(type_num):
        plt.plot(range(len(cc_mean[i])), cc_mean[i], label=f'{experiment_name[i]}')
    plt.xlabel('m')  # 添加横轴标签
    plt.ylabel('cc')  # 添加纵轴标签
    plt.title(f"cc-{dataname}-{time}")
    plt.legend()
    image_save_path = "../output/picture/"#本地保存路径
    plt.savefig(f"{image_save_path}cc-{dataname}-{time}.png")
    plt.show()
def save_results(rmse_base_array, cc_base_array, rmse_mean, rmse_std, cc_mean, experiment_name, budget_set, dataname, time):
    # 本地保存rmse数据
    df = pd.DataFrame(rmse_mean, index=experiment_name, columns=[i for i in range(0, 1+budget_set)])# 创建一个 DataFrame，其中第一列为 'label'，后面是 rmse_base 数值
    df.loc["rmse_base_mean"] = rmse_base_array
    for num, name in enumerate(experiment_name):
        df.loc[f"std-{name}"] = rmse_std[num]
    file_name = f'../output/swstrategy-data/rmse_{dataname}-{time}.xlsx'
    df.to_excel(file_name, index=True)    # 将 DataFrame 写入 Excel 文件

    # 本地保存cc数据
    df = pd.DataFrame(cc_mean, index=experiment_name, columns=[i for i in range(0, 1+budget_set)])# 创建一个 DataFrame，其中第一列为 'label'，后面是 rmse_base 数值
    df.loc["cc_base_mean"] = cc_base_array
    file_name = f'../output/swstrategy-data/cc_{dataname}-{time}.xlsx'
    df.to_excel(file_name, index=True)    # 将 DataFrame 写入 Excel 文件

if __name__ == '__main__':

    # 设置参数
    experiment_name = ["QBC", "sw-QBC", "EMCM", "sw-EMCM", "RD", "sw-RD", "Random"]
    iteration = 100  # 迭代次数
    deviceGPU_set = "cuda:0"
    type_num = len(experiment_name)
    # datanames = ['energy1','concrete', 'IEMOCAP-V', "wine_white",  "wine_red", 'energy2',"airfoil", "autompg", "housing",  "cps"]
    datanames = ['IEMOCAP-V', "wine_white",  "wine_red", 'energy2',"airfoil", "autompg", "housing",  "cps"]
    time = "20250421"
    torch.cuda.set_device(deviceGPU_set)

    for dataname in datanames:
        print(f"====================={dataname}=====================")
        X, y = data_load(dataname)
        budget = 50  # 预算集大小
        minN = X.shape[1]  # 最小训练集大小
        maxN = minN + budget  # 最大训练集大小

        # 单独实验
        # for i in range(type_num):
        #     model = QBC(maxN=maxN, minN=minN, type=i, deviceGPU=deviceGPU_set, random_state=42)
        #     rmse, cc = model.fit(XPool=X, YPool=y)
        #     print(f"{experiment_name[i]}: rmse = {rmse}, cc = {cc}")

        # 训练迭代方法
        rmse_mean, cc_mean, random_array, rmse_std = train(X=X, y=y, experiment_name=experiment_name, maxN=maxN, minN=minN,
                                                iteration=iteration, deviceGPU_set=deviceGPU_set)
        # 用全部数据来训练一个上限
        rmse_base_array, cc_base_array = train_base(X=X, y=y, budget_set=budget, iteration=iteration, random_array=random_array)

        # 绘制结果
        plot_results(rmse_base_array=rmse_base_array, cc_base_array=cc_base_array, rmse_mean=rmse_mean,
                     experiment_name=experiment_name, cc_mean=cc_mean, budget_set=budget, type_num=type_num,dataname=dataname, time=time)
        # 保存结果
        save_results(rmse_base_array=rmse_base_array, cc_base_array=cc_base_array, rmse_mean=rmse_mean,rmse_std =rmse_std,
                     experiment_name=experiment_name, cc_mean=cc_mean, budget_set=budget, dataname=dataname, time=time)
        print(f"=======================end=======================")
