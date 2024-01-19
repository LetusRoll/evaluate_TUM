'''
Description: Python计算TUM格式pose文件的相对平移误差RTE(KITTI数据集里程计评估) 
Version: 1.0
Author: C-Xingyu
Date: 2024-01-16 16:33:04
LastEditors: C-Xingyu
LastEditTime: 2024-01-19 15:18:55
'''

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import transforms3d.euler as euler
import sys

def toCameraCoord(pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0,   0,   1,  0.27],
                          [-1,  0,   0,  0],
                          [0,  -1,   0,  -0.08],
                          [0,   0,   0,  1]])
        inv_R_C2L = np.linalg.inv(R_C2L)            
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot 

def quat_to_rot_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def rotationError( pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d,1.0),-1.0))

def translationError(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx**2+dy**2+dz**2)

def read_tum_file(file_path,toCamera:bool):
    """
    Read a TUM file and return timestamps and poses.
    """
    
    poses_with_timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) != 8:
                continue
            translation = np.array(data[1:4], dtype=float)
            quaternion = Quaternion([float(data[7])] + [float(d) for d in data[4:7]])  # qw qx qy qz

            # 构建齐次变换矩阵
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = quaternion.rotation_matrix
            transformation_matrix[:3, 3] = translation
            if toCamera:
                transformation_matrix = toCameraCoord(transformation_matrix)
            # 添加到位姿列表
            timestamp = float(data[0])
            poses_with_timestamps.append((timestamp, transformation_matrix))
    return poses_with_timestamps



def slerp_interpolation(q1, q2, t):
    """
    在两个四元数之间执行球面线性插值（slerp）。
    输入的四元数为 Quaternion 类型。
    :param q1: 第一个四元数，Quaternion 类型。
    :param q2: 第二个四元数，Quaternion 类型。
    :param t: 插值系数（0 到 1 之间）。
    :return: 插值后的四元数，Quaternion 类型。
    """
    # 确保四元数已经规范化
    q1_normalized = q1.normalised
    q2_normalized = q2.normalised

    # 执行 slerp 插值
    interpolated_quat = Quaternion.slerp(q1_normalized, q2_normalized, t)

    return interpolated_quat



def interpolate_poses(gt_poses_with_timestamps, pred_timestamps):
    """
    根据预测的时间戳对位姿进行插值。
    :param gt_poses_with_timestamps: 包含 (timestamp, pose) 元组的地面真值位姿列表。
    :param pred_timestamps: 预测位姿的时间戳列表。
    :return: 插值后的位姿列表，每个元素为 (timestamp, 4x4 pose matrix)。
    """
    # 提取时间戳和位姿
    gt_timestamps = [item[0] for item in gt_poses_with_timestamps]
    gt_poses = np.array([item[1] for item in gt_poses_with_timestamps])

    # 分离平移和旋转进行插值
    gt_translations = gt_poses[:, :3, 3]
    gt_rotations = [Quaternion(matrix=pose[:3, :3]) for pose in gt_poses]

    # 插值平移
    interp_trans = interp1d(gt_timestamps, gt_translations, axis=0, kind='linear')
    pred_translations = interp_trans(pred_timestamps)

    # 插值旋转
    pred_poses_with_timestamps = []
    for t in pred_timestamps:
        # 寻找最接近的时间戳
        idx1 = np.searchsorted(gt_timestamps, t, side='right') - 1
        idx2 = np.clip(idx1 + 1, 0, len(gt_timestamps) - 1)

        # 获取对应的四元数
        q1 = gt_rotations[idx1]
        q2 = gt_rotations[idx2]

        # 计算插值系数
        factor = (t - gt_timestamps[idx1]) / (gt_timestamps[idx2] - gt_timestamps[idx1]) if gt_timestamps[idx1] != gt_timestamps[idx2] else 0

        # 执行 slerp 插值
        interpolated_quat = slerp_interpolation(q1, q2, factor)

        # 构造插值后的 4x4 变换矩阵
        interpolated_matrix = np.eye(4)
        interpolated_matrix[:3, :3] = interpolated_quat.rotation_matrix
        interpolated_matrix[:3, 3] = pred_translations[np.searchsorted(pred_timestamps, t)]

        # 添加到结果列表
        pred_poses_with_timestamps.append((t, interpolated_matrix))

    return pred_poses_with_timestamps

def find_first_point_index(idx,poses, distance):
    num_points = len(poses)
    second_ind = idx
    dist = 0

    while second_ind + 1 < num_points:
        prev_pose = poses[second_ind][:3, 3]  # 上一个位姿的平移部分
        curr_pose = poses[second_ind + 1][:3, 3]  # 当前位姿的平移部分
        dist += np.linalg.norm(curr_pose - prev_pose)

        if dist >= distance:
            return second_ind + 1

        second_ind += 1

    return -1



def evaluate_relative_poses_at_intervals(gt_poses, pred_poses, intervals):
    """
    Evaluate relative poses at specified distance intervals.
    """
    translation_errors=[]
    rotation_errors =[]
    for distance in intervals:
        trans_error=0
        rot_error=0
        counter=0
        step=10
        for i in range(0,len(pred_poses),step):
            id=find_first_point_index(i,gt_poses,distance)
            if(id<0):
                continue
            first_frame=i
            last_frame=id
            pose_delta_gt = np.dot(np.linalg.inv(gt_poses[first_frame]), gt_poses[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pred_poses[first_frame]), pred_poses[last_frame])
            pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

            # Calculate errors
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)

            trans_error+=t_err
            rot_error+=r_err
            counter=counter+1
        if counter==0:
            print("请减小间距列表中的距离")
            sys.exit(1)
        avg_t_error=trans_error/counter
        avg_r_error=rot_error/counter
        #print("distance: ",distance," m, avarage translation error: ",avg_t_error," avarage rotation error: ",avg_r_error)
        translation_errors.append(avg_t_error/distance)
        rotation_errors.append(avg_r_error/distance)    
    return  translation_errors,rotation_errors
            
    
def plotPath_2D_3( poses_gt, poses_result, plot_path_dir):
    """
    在 XY, XZ 和 YZ 平面上绘制路径。
    :param seq: 序列名称或标识。
    :param poses_gt: 地面真值位姿的列表。
    :param poses_result: 结果位姿的列表。
    :param plot_path_dir: 图像保存路径。
    """
    fontsize_ = 10
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # 提取坐标
    x_gt, y_gt, z_gt = [pose[0, 3] for pose in poses_gt], [pose[1, 3] for pose in poses_gt], [pose[2, 3] for pose in poses_gt]
    x_pred, y_pred, z_pred = [pose[0, 3] for pose in poses_result], [pose[1, 3] for pose in poses_result], [pose[2, 3] for pose in poses_result]

    fig = plt.figure(figsize=(20, 6), dpi=300)
    plot_titles = ["XY Plane", "XZ Plane", "YZ Plane"]
    plot_combinations = [(x_gt, y_gt, x_pred, y_pred), (x_gt, z_gt, x_pred, z_pred), (y_gt, z_gt, y_pred, z_pred)]
    
    for i, plot_data in enumerate(plot_combinations):
        ax = fig.add_subplot(1, 3, i + 1)
        gt_x_data, gt_y_data, pred_x_data, pred_y_data = plot_data
        if poses_gt:
            ax.plot(gt_x_data, gt_y_data, style_gt, label="Ground Truth")
        ax.plot(pred_x_data, pred_y_data, style_pred, label="Ours")
        ax.plot(0, 0, style_O, label='Start Point')
        ax.set_xlabel('x (m)' if i != 2 else 'y (m)', fontsize=fontsize_)
        ax.set_ylabel('y (m)' if i == 0 else 'z (m)', fontsize=fontsize_)
        ax.legend(loc="upper right", prop={'size': fontsize_})
        ax.set_title(plot_titles[i])
        ax.axis('equal')  # 确保 XY, XZ, YZ 平面比例一致

    png_title = f"path_2D_3"
    plt.savefig(f"{plot_path_dir}/{png_title}.png", bbox_inches='tight', pad_inches=0.1)
   # pdf = matplotlib.backends.backend_pdf.PdfPages(f"{plot_path_dir}/{png_title}.pdf")        
    fig.tight_layout()
    #pdf.savefig(fig)  
    plt.close()

def plot_xyz( poses_ref, poses_pred, plot_path_dir):
    def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
        """
        在轴数组上基于 xyz 坐标绘制路径/轨迹。
        """
        x = range(0, len(positions_xyz))
        xlabel = "index"
        ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
        for i in range(0, 3):
            axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
            axarr[i].set_ylabel(ylabels[i])
            axarr[i].legend(loc="upper right", frameon=True)
        axarr[2].set_xlabel(xlabel)
        if title:
            axarr[0].set_title('XYZ')

    fig, axarr = plt.subplots(3, sharex="col", figsize=(20, 10))
    
    # 从位姿列表中提取 XYZ 坐标
    pred_xyz = np.array([p[:3, 3] for p in poses_pred])
    traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)

    if poses_ref:
        ref_xyz = np.array([p[:3, 3] for p in poses_ref])
        traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)

    name = f"xyz"
    plt.savefig(f"{plot_path_dir}/{name}.png", bbox_inches='tight', pad_inches=0.1)
   # pdf = matplotlib.backends.backend_pdf.PdfPages(f"{plot_path_dir}/{name}.pdf")
    fig.tight_layout()
   # pdf.savefig(fig)
    #pdf.close()
    plt.close()  

def plot_rpy( poses_ref, poses_pred, plot_path_dir, axes='szxy'):
    def traj_rpy(axarr, orientations_euler, style='-', color='black', title="", label="", alpha=1.0):
        """
        绘制轨迹的欧拉RPY角到轴上。
        """
        x = range(0, len(orientations_euler))
        xlabel = "index"
        ylabels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
        for i in range(0, 3):
            axarr[i].plot(x, np.rad2deg(orientations_euler[:, i]), style,
                          color=color, label=label, alpha=alpha)
            axarr[i].set_ylabel(ylabels[i])
            axarr[i].legend(loc="upper right", frameon=True)
        axarr[2].set_xlabel(xlabel)
        if title:
            axarr[0].set_title('PRY')

    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=(20, 10))

    # 从位姿列表中提取RPY角度
    pred_rpy = np.array([euler.mat2euler(p, axes=axes) for p in poses_pred])
    traj_rpy(axarr_rpy, pred_rpy, '-', 'b', title='RPY', label='Ours', alpha=1.0)

    if poses_ref:
        ref_rpy = np.array([euler.mat2euler(p, axes=axes) for p in poses_ref])
        traj_rpy(axarr_rpy, ref_rpy, '-', 'r', label='GT', alpha=1.0)

    name = f"rpy"
    plt.savefig(f"{plot_path_dir}/{name}.png", bbox_inches='tight', pad_inches=0.1)
   # pdf = matplotlib.backends.backend_pdf.PdfPages(f"{plot_path_dir}/{name}.pdf")
    fig_rpy.tight_layout()
   # pdf.savefig(fig_rpy)
    #pdf.close()
    plt.close()



def plotError_segment(intervals, avg_trans_errs, avg_rot_errs,plot_error_dir):
        '''
            avg_segment_errs: dict [100: err, 200: err...]
        '''
        fontsize_ = 15
        plot_y_t = []
        plot_y_r = []
        plot_x = []
        for idx in range(len(intervals)):
            
            plot_x.append(intervals[idx])
            plot_y_t.append(avg_trans_errs[idx]*100)
            plot_y_r.append(avg_rot_errs[idx]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=300)
        plt.subplot(1,2,1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Translation Error (%)',fontsize=fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize=fontsize_)
        png_title = "error_seg"
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        #plt.show()
    
def plotPath_3D(poses_gt, poses_result, plot_path_dir):
    """
    在 3D 空间中绘制路径
    :param poses_gt: 地面真值位姿的列表。
    :param poses_result: 预测位姿的列表。
    :param plot_path_dir: 图像保存路径。
    """
    fontsize_ = 8
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    if poses_gt:
        gt_x = [pose[0, 3] for pose in poses_gt]
        gt_y = [pose[1, 3] for pose in poses_gt]
        gt_z = [pose[2, 3] for pose in poses_gt]
        ax.plot(gt_x, gt_y, gt_z, style_gt, label="Ground Truth")

    result_x = [pose[0, 3] for pose in poses_result]
    result_y = [pose[1, 3] for pose in poses_result]
    result_z = [pose[2, 3] for pose in poses_result]
    ax.plot(result_x,result_y , result_z, style_pred, label="Ours")

    # 标记起始点
    ax.plot([0], [0], [0], style_O, label='Start Point')

    # 设置坐标轴限制
    ax.set_xlabel('x (m)', fontsize=fontsize_)
    ax.set_ylabel('y (m)', fontsize=fontsize_)
    ax.set_zlabel('z (m)', fontsize=fontsize_)
    ax.legend()
    ax.view_init(elev=20., azim=-35)

    # 保存图像
    png_title = "path_3D"
    plt.savefig(f"{plot_path_dir}/{png_title}.png", bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()
def plotPath_XZ(poses_gt, poses_result, plot_path_dir):
    """
    在 XZ 平面上绘制路径。
    :param poses_gt: 地面真值位姿的列表。
    :param poses_result: 结果位姿的列表。
    :param plot_path_dir: 图像保存路径。
    """
    fontsize_ = 10
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # 提取坐标
    x_gt = [pose[0, 3] for pose in poses_gt] if poses_gt else []
    z_gt = [pose[2, 3] for pose in poses_gt] if poses_gt else []
    x_pred = [pose[0, 3] for pose in poses_result]
    z_pred = [pose[2, 3] for pose in poses_result]

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    if poses_gt:
        ax.plot(x_gt, z_gt, style_gt, label="Ground Truth")
    ax.plot(x_pred, z_pred, style_pred, label="Ours")
    ax.plot(0, 0, style_O, label='Start Point')
    
    ax.set_xlabel('x (m)', fontsize=fontsize_)
    ax.set_ylabel('z (m)', fontsize=fontsize_)
    ax.legend(loc="upper right", prop={'size': fontsize_})
    ax.set_title("XZ Plane")
    ax.axis('equal')

    png_title = f"path_XZ"
    plt.savefig(f"{plot_path_dir}/{png_title}.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    # Example usage
    gt_file = '/home/cxy/evaluate_TUM/gt05_tum.txt'
    pred_file = '/home/cxy/evaluate_TUM/pose.txt'


    gt_poses_with_timestamps = read_tum_file(gt_file,toCamera=False)
    pred_poses_with_timestamps = read_tum_file(pred_file,toCamera=True)#输入是LiDAR的位姿

    # 裁剪 pred_poses 使其时间范围不超过 gt_poses
    while gt_poses_with_timestamps[-1][0] < pred_poses_with_timestamps[-1][0]:
        pred_poses_with_timestamps = pred_poses_with_timestamps[:-1]

    while gt_poses_with_timestamps[0][0] > pred_poses_with_timestamps[0][0]:
        pred_poses_with_timestamps = pred_poses_with_timestamps[1:]
    #如果需要，可以从列表中提取独立的时间戳和位姿列表
    pred_timestamps = [item[0] for item in pred_poses_with_timestamps]
    interpolated_gt_poses = interpolate_poses(gt_poses_with_timestamps, pred_timestamps)


    print(len(interpolated_gt_poses))
    print(len(pred_poses_with_timestamps))

    intervals = [100, 200,300,400,500,600,700,800]  # Define distance intervals (in meters)
    gt_poses=[pose_matrix for _, pose_matrix in interpolated_gt_poses]
    pred_poses = [item[1] for item in pred_poses_with_timestamps]

    # gt_filename = "/home/cxy/gt.txt"
    # odom_filename = "/home/cxy/pose.txt"

    # # 使用 with 语句打开文件，确保文件会被正确关闭
    # with open(gt_filename, 'w') as file:
    #     # 遍历列表中的每个元素
    #     for item in gt_poses:
    #         translation = item[:3, 3]
    #         translation_str = ' '.join(map(str, translation))
    #         # 写入文件
    #         file.write(translation_str + "\n")
    # with open(odom_filename, 'w') as file:
    #     # 遍历列表中的每个元素
    #     for item in pred_poses:
    #         translation = item[:3, 3]
    #         translation_str = ' '.join(map(str, translation))
    #         # 写入文件
    #         file.write(translation_str + "\n")

    translation_errors, rotation_errors = evaluate_relative_poses_at_intervals(gt_poses,pred_poses, intervals)

    # Output errors for each interval
    for i in range(len(intervals)):
        print(f"Interval {intervals[i]}m: Avg Translation Error for 100m: {translation_errors[i]*100}")
        print(f"Interval {intervals[i]}m: Avg Rotation Error for °/m: {rotation_errors[i]*180/3.14159}")

    # Example usage
    plotPath_3D(gt_poses,pred_poses,"/home/cxy")
    plotPath_2D_3(gt_poses,pred_poses,"/home/cxy")
    plot_xyz(gt_poses,pred_poses,"/home/cxy")
    plot_rpy(gt_poses,pred_poses,"/home/cxy")
    plotError_segment(intervals,translation_errors,rotation_errors,"/home/cxy")
