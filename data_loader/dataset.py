"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


class Dataset:

    def __init__(self, mode, t_his, t_pred, actions='all'):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def Print3D(self, num_frame, point, arms, rightHand, leftHand, legs, body):
        # 求坐标最大值
        xmax = np.max(point[0, :, :])
        xmin = np.min(point[0, :, :])
        ymax = np.max(point[1, :, :])
        ymin = np.min(point[1, :, :])
        zmax = np.max(point[2, :, :])
        zmin = np.min(point[2, :, :])

        n = 0  # 从第n帧开始展示
        m = num_frame  # 到第m帧结束，n<m<row
        plt.figure()
        plt.ion()
        for i in range(n, m):
            plt.cla()  # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

            plot3D = plt.subplot(projection='3d')
            plot3D.view_init(15, -30)  # 改变视角

            Expan_Multiple = 1.4  # 坐标扩大倍数，绘图较美观

            # 画出两个body所有关节
            plot3D.scatter(point[0, i, :] * Expan_Multiple, point[1, i, :] * Expan_Multiple, point[2, i, :],
                           c='red', s=40.0)  # c: 颜色;  s: 大小

            # 连接第一个body的关节，形成骨骼
            plot3D.plot(point[0, i, arms] * Expan_Multiple, point[1, i, arms] * Expan_Multiple, point[2, i, arms],
                        c='green', lw=2.0)
            plot3D.plot(point[0, i, rightHand] * Expan_Multiple, point[1, i, rightHand] * Expan_Multiple,
                        point[2, i, rightHand], c='green', lw=2.0)  # c: 颜色;  lw: 线条宽度
            plot3D.plot(point[0, i, leftHand] * Expan_Multiple, point[1, i, leftHand] * Expan_Multiple,
                        point[2, i, leftHand], c='green', lw=2.0)
            plot3D.plot(point[0, i, legs] * Expan_Multiple, point[1, i, legs] * Expan_Multiple, point[2, i, legs],
                        c='green', lw=2.0)
            plot3D.plot(point[0, i, body] * Expan_Multiple, point[1, i, body] * Expan_Multiple, point[2, i, body],
                        c='green', lw=2.0)

            # 连接第二个body的关节，形成骨骼
            # plot3D.plot(point[0, i, arms, 1] * Expan_Multiple, point[1, i, arms, 1] * Expan_Multiple, point[2, i, arms, 1],
            #             c='green', lw=2.0)
            # plot3D.plot(point[0, i, rightHand, 1] * Expan_Multiple, point[1, i, rightHand, 1] * Expan_Multiple,
            #             point[2, i, rightHand, 1], c='green', lw=2.0)
            # plot3D.plot(point[0, i, leftHand, 1] * Expan_Multiple, point[1, i, leftHand, 1] * Expan_Multiple,
            #             point[2, i, leftHand, 1], c='green', lw=2.0)
            # plot3D.plot(point[0, i, legs, 1] * Expan_Multiple, point[1, i, legs, 1] * Expan_Multiple, point[2, i, legs, 1],
            #             c='green', lw=2.0)
            # plot3D.plot(point[0, i, body, 1] * Expan_Multiple, point[1, i, body, 1] * Expan_Multiple, point[2, i, body, 1],
            #             c='green', lw=2.0)

            plot3D.text(xmax - 0.3, ymax + 1.1, zmax + 0.3, 'frame: {}/{}'.format(i, num_frame - 1))  # 文字说明
            plot3D.set_xlim3d(xmin - 0.5, xmax + 0.5)  # x坐标范围
            plot3D.set_ylim3d(ymin - 0.3, ymax + 0.3)  # y坐标范围
            plot3D.set_zlim3d(zmin - 0.3, zmax + 0.3)  # z坐标范围
            plt.pause(0.001)  # 停顿延时

        plt.ioff()
        plt.show()

    def sample(self):
        subject = np.random.choice(self.subjects)
        '''
        self.subjects = {'train': [1, 5, 6, 7, 8],
                         'test': [9, 11]}
        '''
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]  # 随机取动作
        # seq = dict_s['WalkDog']
        fr_start = np.random.randint(seq.shape[0] - self.t_total)  # t_total= t_his + t_pred
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]  # shape : T V C
        # data = torch.tensor(traj)
        # data = data.permute(2,0,1).contiguous()  # N T V C -> N C T V
        # data = data.numpy()
        # arms = [27, 26, 25, 13, 17, 18, 19]  # 23 <-> 11 <-> 10 ...
        # rightHand = [27]  # 11 <-> 24
        # leftHand = [19]  # 7 <-> 22
        # legs = [3, 2, 1, 0, 6, 7, 8]  # 19 <-> 18 <-> 17 ...
        # body = [0, 12, 13, 14, 15]  # 3 <-> 2 <-> 20 ...
        # self.Print3D(100, point=data, arms=arms, rightHand=rightHand, leftHand=leftHand, legs=legs, body=body)
        return traj[None, ...]  # shape: 1 T V C

    def sample_all_action(self):
        # subject = np.random.choice(self.subjects)
        dict_s = self.data['S9']

        action_list = []
        sample = []

        for i in range(0, len(list(dict_s.keys()))):
            type = list(dict_s.keys())[i].split(' ')[0]
            if type == 'Discussion':
                type = 'Discussion 1'
            action_list.append(type)

        action_list = list(set(action_list))

        for i in range(0, len(action_list)):
            action = action_list[i]
            seq = dict_s[action]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start: fr_end]
            sample.append(traj[None, ...])

        # 15 -> 30
        # for i in range(0, len(action_list)):
        #     action = action_list[i]
        #     seq = dict_s[action]
        #     fr_start = np.random.randint(seq.shape[0] - self.t_total)
        #     fr_end = fr_start + self.t_total
        #     traj = seq[fr_start: fr_end]
        #     sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample

    def sample_iter_action(self, action_category, dataset_type):
        # subject = np.random.choice(self.subjects)
        if dataset_type == 'h36m':
            dict_s = self.data['S9']
        elif dataset_type == 'humaneva':
            dict_s = self.data['Validate/S2']
        else:
            raise
        # dict_s = self.data['S9']
        sample = []

        action = action_category
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample

    def prepare_iter_action(self, dataset_type):
        # subject = np.random.choice(self.subjects)
        if dataset_type == 'h36m':
            dict_s = self.data['S9']
        elif dataset_type == 'humaneva':
            dict_s = self.data['Validate/S2']
        else:
            raise
        # dict_s = self.data['S9']

        action_list = []
        sample = []

        for i in range(0, len(list(dict_s.keys()))):
            # type = list(dict_s.keys())[i].split(' ')[0]
            type = list(dict_s.keys())[i]
            if type == 'Discussion':
                type = 'Discussion 1'
            action_list.append(type)

        action_list = list(set(action_list))
        return action_list

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for i in range(num_samples // batch_size):  # num sample:数据总量
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug is True:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(sample.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    sample[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    sample[..., 0] = - sample[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    sample[..., 1] = - sample[..., 1]
            yield sample

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj
