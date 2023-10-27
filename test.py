import torch
import numpy as np
# a=torch.randn(50,1,10,75)
# b=torch.randn(1,3,10,75)
# diff_multi = a - b
# dist = torch.linalg.norm(diff_multi, dim=1)
# print(dist.shape)
# mmfde, _ = dist[:, :-1, -1].min(dim=0)

a=torch.randn(4,125,3*16)
m = torch.randn(125,125)
b=torch.matmul(m[:10],a)
# print(a)
print(b.shape)

# import numpy as np
# data = np.load('data/data_3d_h36m_test.npz',allow_pickle=True)#读取npz文件
# print(data.files)
# for i in data['data']:
#     print(i)
    #print(data['data'].item()[i].keys())
#print(data['positions_3d'].item()['S6'].keys())#查看文件中包含的数组名

# def get_dct_matrix(N, is_torch=True):
#     dct_m = np.eye(N)  # 创建一个N×N的单位矩阵dct_m
#     for k in np.arange(N):
#         for i in np.arange(N):
#             w = np.sqrt(2 / N)  # 第k行第i列的权重系数
#             if k == 0:
#                 w = np.sqrt(1 / N)
#             dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)  # 公式 w * cos(π(i+1/2)k/N)
#     idct_m = np.linalg.inv(dct_m)  # 计算逆dct矩阵
#     if is_torch:
#         dct_m = torch.from_numpy(dct_m)
#         idct_m = torch.from_numpy(idct_m)
#     return dct_m, idct_m
#
#
# dct_m, idct_m=get_dct_matrix(25)
# print(dct_m.shape)

# import numpy as np
# noise_steps=1000
# ddim_timesteps=100
# ddim_timestep_seq = np.asarray(
#     list(range(0, noise_steps, noise_steps // ddim_timesteps))) + 1
# ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
#
# print(ddim_timestep_seq)
# print(ddim_timestep_prev_seq)



