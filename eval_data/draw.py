import sys, os
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from cyber_py3 import cyber
from cyber_py3 import record

from modules.perception.proto import perception_obstacle_pb2
from modules.prediction.proto import feature_pb2
from modules.common.proto import pnc_point_pb2
from google.protobuf import text_format

class Visualizer():
    def __init__(self, cfg={}):
        self.root = "/apollo/eval_data"
        if "history_length" in cfg:
            self.history_length = cfg["history_length"]
        else:
            self.history_length = 20
        if "prediction_length" in cfg:
            self.prediction_length = cfg["prediction_length"]
        else:
            self.prediction_length = 30
        self.history_traj = np.zeros((self.history_length,2))
        self.modified_history_traj = np.zeros((self.history_length, 2))
        self.gt_traj = np.zeros((self.prediction_length,2))
        self.record_path = "/apollo/modules/prediction/eval_data/test2.record"

        freader = record.RecordReader(self.record_path)
        obstacle_msg_cnt = 0
        for topic, msg, _, timestamp in freader.read_messages():
            if topic == "/apollo/perception/obstacles":
                parsed = perception_obstacle_pb2.PerceptionObstacles()
                parsed.ParseFromString(msg)
                msg = parsed

                x = msg.perception_obstacle[0].position.x
                y = msg.perception_obstacle[0].position.y
                if obstacle_msg_cnt < self.history_length:
                    self.history_traj[obstacle_msg_cnt,0] = x
                    self.history_traj[obstacle_msg_cnt,1] = y
                elif obstacle_msg_cnt < self.history_length + self.prediction_length:
                    self.gt_traj[obstacle_msg_cnt-self.history_length,0] = x
                    self.gt_traj[obstacle_msg_cnt-self.history_length,1] = y
                else:
                    break

                obstacle_msg_cnt += 1

        with open("/apollo/eval_data/trajectories/history.pb.txt", 'r') as f:
            traj = feature_pb2.Trajectory()
            text_format.Parse(f.read(), traj)
            for i in range(self.history_length):
                self.modified_history_traj[i,0] = traj.trajectory_point[i].path_point.x
                self.modified_history_traj[i,1] = traj.trajectory_point[i].path_point.y


    def set_figure(self):
        fig, ax = plt.subplots(figsize=(15,15))
        ax.set_xlim(587470,587500)
        ax.set_ylim(4140685,4140715)
        ax.plot([587470.62, 587470.62 + (587480.36 - 587470.62) * 3], [4140711.88, 4140711.88 + (4140697.39 - 4140711.88) * 3], 'k')
        ax.plot([587473.07, 587473.07 + (587483.32 - 587473.07) * 3], [4140714.42, 4140714.42 + (4140699.51 - 4140714.42) * 3], 'k')
        ax.plot([587475.53, 587475.53 + (587486.20 - 587475.53) * 3], [4140716.28, 4140716.28 + (4140701.63 - 4140716.28) * 3], 'k')
        return fig, ax


    def radial_heatmap(self, rad, a, data, label):
        fig = plt.figure()
        ax = Axes3D(fig)
        r, th = np.meshgrid(rad, a)

        plt.subplot(projection="polar")
        plt.pcolormesh(th, r, data, cmap = 'inferno')
        plt.plot(a, r, ls='none', color = 'k') 
        plt.grid()
        plt.colorbar()
        plt.savefig(os.path.join(self.root, '{}.png'.format(label)))


    def mlp(self):
        my_data = np.genfromtxt(os.path.join(self.root, "output/output1"), delimiter=',')
        rad = np.linspace(0.1, 1.0, 10)
        a = np.linspace(0, 2 * np.pi, 12)
        
        self.radial_heatmap(rad, a, my_data[:,3].reshape((10,12)).T, "mlp_heatmap")
        # self.radial_heatmap(rad, a, (my_data[:,4] == 0).reshape((10,12)).T, "mlp_bin_heatmap")
        self.radial_heatmap(rad, a, my_data[:,4].reshape((10,12)).T, "mlp_heatmap_2")


    def lstm(self):
        predicted_traj = np.zeros((10,12,6,2))
        for d in range(10):
            for t in range(12):
                try:
                    with open(os.path.join(self.root, "trajectories/{}_{}_0_1.pb.txt".format(d+1, t+1)), 'r') as f:
                        traj = feature_pb2.Trajectory()
                        text_format.Parse(f.read(), traj)
                        for i in range(self.prediction_length):
                            predicted_traj[d,t,i,0] = traj.trajectory_point[i].path_point.x
                            predicted_traj[d,t,i,1] = traj.trajectory_point[i].path_point.y
                except Exception as e:
                    print(e)

        # ADE & FDE
        extended_gt_traj = np.tile(self.gt_traj, (10,12, 1, 1))
        FDE = np.sum(np.power(extended_gt_traj[:,:,-1,:] - predicted_traj[:,:,-1,:], 2), axis=2)
        # print(FDE)
        ADE = np.mean(np.sum(np.power(extended_gt_traj - predicted_traj, 2), axis=3), axis=2)
        # print(ADE)

        rad = np.linspace(0.1, 1.0, 10)
        a = np.linspace(0, 2 * np.pi, 12)
        self.radial_heatmap(rad, a, FDE.T, "lstm_fde")
        self.radial_heatmap(rad, a, ADE.T, "lstm_ade")

        fig, ax = self.set_figure()
        ax.plot(self.gt_traj[:,0], self.gt_traj[:,1], 'bo-')
        ax.plot(self.history_traj[:,0], self.history_traj[:,1], 'bo-')
        for d in range(10):
            for t in range(12):
                ax.plot(predicted_traj[d,t,:,0], predicted_traj[d,t,:,1], 'o:')
        fig.savefig(os.path.join(self.root, "traj.png"))


    def single_traj_lstm(self, filename, label="traj"):
        predicted_traj = np.zeros((self.prediction_length,2))
        with open(filename, 'r') as f:
            traj = feature_pb2.Trajectory()
            text_format.Parse(f.read(), traj)
            for i in range(self.prediction_length):
                predicted_traj[i,0] = traj.trajectory_point[i].path_point.x
                predicted_traj[i,1] = traj.trajectory_point[i].path_point.y
        fig, ax = self.set_figure()
        ax.plot(self.gt_traj[:,0], self.gt_traj[:,1], 'bo:')
        ax.plot(self.history_traj[:,0], self.history_traj[:,1], 'bo-')
        ax.plot(predicted_traj[:,0], predicted_traj[:,1], 'ro:')
        ax.plot(self.modified_history_traj[:,0], self.modified_history_traj[:,1], 'ro-')
        fig.savefig(os.path.join(self.root, "{}.png".format(label)))

if __name__ == "__main__":
    V = Visualizer()
    # V.mlp()
    # V.lstm()
    V.single_traj_lstm("/apollo/eval_data/trajectories/evaluate.pb.txt", "traj1")
    V.single_traj_lstm("/apollo/eval_data/trajectories/predict.pb.txt", "traj2")