#!/usr/bin/env python3
"""
运行 LiDAR 仿真并截图
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 尝试导入 sklearn
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class Box:
    def __init__(self, center, size, color='blue'):
        self.center = np.array(center)
        self.size = np.array(size)
        self.color = color
        self.min_bound = self.center - self.size / 2
        self.max_bound = self.center + self.size / 2

    def intersect(self, ray_origin, ray_dir):
        inv_dir = 1.0 / (ray_dir + 1e-9)
        t1 = (self.min_bound - ray_origin) * inv_dir
        t2 = (self.max_bound - ray_origin) * inv_dir
        tmin = np.maximum(np.minimum(t1, t2), 0.0)
        tmax = np.minimum(np.maximum(t1, t2), 1e9)
        t_enter = np.max(tmin)
        t_exit = np.min(tmax)
        if t_exit >= t_enter:
            return t_enter
        return np.inf

class Environment3D:
    def __init__(self):
        self.obstacles = []
        self.obstacles.append(Box([0, 0, -1], [100, 100, 2], color='gray'))
        self.obstacles.append(Box([10, 5, 2], [2, 2, 4], color='red'))
        self.obstacles.append(Box([15, -5, 3], [3, 6, 6], color='green'))
        self.obstacles.append(Box([20, 0, 1], [1, 1, 2], color='orange'))
        self.obstacles.append(Box([40, 0, 10], [2, 40, 20], color='gray'))

    def check_collision(self, ray_origin, ray_dir):
        closest_t = np.inf
        for obs in self.obstacles:
            t = obs.intersect(ray_origin, ray_dir)
            if t < closest_t:
                closest_t = t
        return closest_t

class Lidar3D:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.range = 50.0
        self.fov_v = (-15, 15)
        self.fov_h = (0, 360)
        self.res_v = 32
        self.res_h = 80
        self.ray_dirs = self._generate_ray_dirs()

    def _generate_ray_dirs(self):
        v_angles = np.linspace(np.radians(self.fov_v[0]), np.radians(self.fov_v[1]), self.res_v)
        h_angles = np.linspace(np.radians(self.fov_h[0]), np.radians(self.fov_h[1]), self.res_h)
        ray_dirs = []
        for v in v_angles:
            for h in h_angles:
                dx = np.cos(v) * np.cos(h)
                dy = np.cos(v) * np.sin(h)
                dz = np.sin(v)
                ray_dirs.append(np.array([dx, dy, dz]))
        return np.array(ray_dirs)

    def scan(self, env):
        points = []
        for d in self.ray_dirs:
            noisy_d = d + np.random.normal(0, 0.005, 3)
            noisy_d = noisy_d / np.linalg.norm(noisy_d)
            t = env.check_collision(self.position, noisy_d)
            if t < self.range:
                noisy_t = t + np.random.normal(0, 0.05)
                if noisy_t > 0:
                    point = self.position + noisy_d * noisy_t
                    points.append(point)
        return np.array(points)

def cluster_points(points):
    if not HAS_SKLEARN or len(points) == 0:
        return np.zeros(len(points))
    clustering = DBSCAN(eps=2.0, min_samples=3).fit(points)
    return clustering.labels_

def draw_box(ax, box):
    c = box.center
    s = box.size / 2
    x = [c[0]-s[0], c[0]+s[0]]
    y = [c[1]-s[1], c[1]+s[1]]
    z = [c[2]-s[2], c[2]+s[2]]
    ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], [z[0], z[0], z[0], z[0], z[0]], color=box.color, alpha=0.5)
    ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], [z[1], z[1], z[1], z[1], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[0], x[0]], [y[0], y[0]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[1], x[1]], [y[0], y[0]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[1], x[1]], [y[1], y[1]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[0], x[0]], [y[1], y[1]], [z[0], z[1]], color=box.color, alpha=0.5)

def main():
    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'docs', 'images')
    os.makedirs(output_dir, exist_ok=True)

    env = Environment3D()
    lidar = Lidar3D([5, 0, 2])

    # 创建图形
    fig = plt.figure(figsize=(16, 7))

    # 扫描
    points = lidar.scan(env)

    # === 图1: 仿真环境视图 ===
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Simulation Environment", fontsize=14)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 绘制环境
    for obs in env.obstacles:
        draw_box(ax1, obs)

    # 绘制 LiDAR 位置
    ax1.scatter([lidar.position[0]], [lidar.position[1]], [lidar.position[2]],
                color='black', s=100, marker='^', label='LiDAR')

    # 绘制扫描射线（部分）
    for i, d in enumerate(lidar.ray_dirs[::200]):  # 每200条画一条
        end_point = lidar.position + d * 10
        ax1.plot([lidar.position[0], end_point[0]],
                [lidar.position[1], end_point[1]],
                [lidar.position[2], end_point[2]],
                'r-', alpha=0.2, linewidth=0.5)

    ax1.set_xlim(-5, 45)
    ax1.set_ylim(-20, 20)
    ax1.set_zlim(0, 15)
    ax1.legend()

    # === 图2: 点云视图 ===
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("LiDAR Point Cloud", fontsize=14)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    if len(points) > 0:
        labels = cluster_points(points)
        unique_labels = set(labels)
        colors = plt.cm.jet(np.linspace(0, 1, max(len(unique_labels), 1)))

        c_array = []
        for l in labels:
            if l == -1:
                c_array.append([0, 0, 0, 1])
            else:
                c_array.append(colors[l % len(colors)])

        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=c_array, s=3)

        # 添加统计信息
        ax2.text2D(0.05, 0.95, f"Points: {len(points)}\nClusters: {len(unique_labels)}",
                   transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    ax2.set_xlim(-5, 45)
    ax2.set_ylim(-20, 20)
    ax2.set_zlim(0, 15)

    plt.tight_layout()

    # 保存主截图
    screenshot_path = os.path.join(output_dir, 'main-interface.png')
    fig.savefig(screenshot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {screenshot_path}")
    plt.close()

    # === 生成不同视角的截图 ===
    views = [
        ('top-view', (90, -90), "Top View"),
        ('front-view', (0, 0), "Front View"),
        ('side-view', (0, -90), "Side View"),
        ('isometric-view', (30, -45), "Isometric View"),
    ]

    for filename, (elev, azim), title in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=14)

        # 绘制环境和点云
        for obs in env.obstacles:
            draw_box(ax, obs)
        ax.scatter([lidar.position[0]], [lidar.position[1]], [lidar.position[2]],
                   color='black', s=100, marker='^')

        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c_array, s=3, alpha=0.7)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-5, 45)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 15)

        path = os.path.join(output_dir, f'{filename}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
        plt.close()

    # === LiDAR 模型特写 ===
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("LiDAR Model", fontsize=14)

    # 绘制简化的 LiDAR 模型
    # 底座
    theta = np.linspace(0, 2*np.pi, 32)
    z_cyl = np.linspace(0, 0.3, 5)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = 0.3 * np.cos(theta_grid) + lidar.position[0]
    y_cyl = 0.3 * np.sin(theta_grid) + lidar.position[1]
    z_cyl_shifted = z_grid + lidar.position[2] - 0.3
    ax.plot_surface(x_cyl, y_cyl, z_cyl_shifted, color='gray', alpha=0.8)

    # 顶部旋转头
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x_sphere = 0.2 * np.outer(np.cos(u), np.sin(v)) + lidar.position[0]
    y_sphere = 0.2 * np.outer(np.sin(u), np.sin(v)) + lidar.position[1]
    z_sphere = 0.2 * np.outer(np.ones(np.size(u)), np.cos(v)) + lidar.position[2] + 0.2
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='darkblue', alpha=0.8)

    # 绘制扫描射线
    for i, d in enumerate(lidar.ray_dirs[::100]):
        end_point = lidar.position + d * 8
        ax.plot([lidar.position[0], end_point[0]],
                [lidar.position[1], end_point[1]],
                [lidar.position[2], end_point[2]],
                'r-', alpha=0.3, linewidth=0.5)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-2, 10)
    ax.view_init(elev=20, azim=-60)

    path = os.path.join(output_dir, 'lidar-model.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()

    # === 点云详情 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Point Cloud Detail", fontsize=14)

    if len(points) > 0:
        # 按高度着色
        colors_z = points[:, 2]
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=colors_z, cmap='viridis', s=5, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='Height (Z)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5, 45)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 15)

    path = os.path.join(output_dir, 'point-cloud-detail.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()

    print(f"\n所有截图已保存到: {output_dir}")

if __name__ == "__main__":
    main()
