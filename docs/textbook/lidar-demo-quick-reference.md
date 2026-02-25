# LiDAR 仿真快速参考手册

## 核心概念速查

### 射线投射原理

```
射线定义: P(t) = origin + t * direction
- origin: 射线起点（LiDAR 位置）
- direction: 射线方向（单位向量）
- t: 参数，t > 0 表示射线正方向
```

### 坐标转换

**球坐标 → 直角坐标**：

```python
# 已知：距离 d，水平角 h，垂直角 v（弧度）
x = d * cos(v) * cos(h)
y = d * cos(v) * sin(h)
z = d * sin(v)
```

---

## 关键参数

### LiDAR 参数

| 参数 | 变量名 | 默认值 | 说明 |
|------|--------|--------|------|
| 最大探测距离 | `range` | 50.0 | 米 |
| 垂直视场角 | `fov_v` | (-15, 15) | 度 |
| 水平视场角 | `fov_h` | (0, 360) | 度 |
| 垂直分辨率 | `res_v` | 32 | 线数 |
| 水平分辨率 | `res_h` | 80 | 点数/圈 |

### 计算公式

```python
# 总射线数
total_rays = res_v * res_h

# 垂直角度间隔
v_step = (fov_v[1] - fov_v[0]) / (res_v - 1)

# 水平角度间隔
h_step = (fov_h[1] - fov_h[0]) / res_h
```

### 噪声参数

| 类型 | 默认值 | 代码位置 |
|------|--------|---------|
| 方向噪声 | σ = 0.005 | `noisy_d = d + np.random.normal(0, 0.005, 3)` |
| 距离噪声 | σ = 0.05 | `noisy_t = t + np.random.normal(0, 0.05)` |

---

## AABB 碰撞检测（Slab 方法）

### 算法原理

```
1. 对于每个轴（X, Y, Z）：
   - 计算 t1（进入 Slab 的时间）
   - 计算 t2（离开 Slab 的时间）

2. t_enter = max(t1_x, t1_y, t1_z)
   t_exit = min(t2_x, t2_y, t2_z)

3. 如果 t_enter <= t_exit，则相交
```

### 代码实现

```python
def intersect(ray_origin, ray_dir, min_bound, max_bound):
    inv_dir = 1.0 / (ray_dir + 1e-9)  # 避免除零

    t1 = (min_bound - ray_origin) * inv_dir
    t2 = (max_bound - ray_origin) * inv_dir

    tmin = np.maximum(np.minimum(t1, t2), 0.0)
    tmax = np.minimum(np.maximum(t1, t2), 1e9)

    t_enter = np.max(tmin)
    t_exit = np.min(tmax)

    return t_enter if t_exit >= t_enter else np.inf
```

---

## DBSCAN 聚类

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eps` | 2.0 | 邻域半径 |
| `min_samples` | 3 | 最小样本数 |

### 代码实现

```python
from sklearn.cluster import DBSCAN

def cluster_points(points, eps=2.0, min_samples=3):
    if len(points) == 0:
        return np.zeros(0)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_
```

### 标签含义

- `label >= 0`: 属于该编号的聚类
- `label == -1`: 噪声点（不属于任何聚类）

---

## 版本对比

| 特性 | 基础版 | 交互版 | Open3D版 | 双窗口版 |
|------|-------|-------|---------|---------|
| 文件 | `lidar_simulation.py` | `lidar_simulation_interactive.py` | `lidar_simulation_open3d.py` | `lidar_simulation_dual.py` |
| 渲染引擎 | Matplotlib | Matplotlib | Open3D GUI | Open3D |
| 交互方式 | 自动移动 | 键盘控制 | 键盘控制 | 键盘控制 |
| 默认分辨率 | 15×60 | 32×80 | 64×200 | 64×200 |
| 帧率 | ~10 FPS | ~20 FPS | ~100 FPS | ~60 FPS |
| GPU 加速 | 否 | 否 | 是 | 是 |
| 依赖 | numpy, matplotlib | numpy, matplotlib | numpy, open3d | numpy, open3d |

---

## 常用代码片段

### 创建 LiDAR

```python
lidar = Lidar3D([0, 0, 2])  # 初始位置 (x, y, z)
```

### 创建环境

```python
env = Environment3D()
# 添加自定义障碍物
env.obstacles.append(Box([x, y, z], [w, h, d], 'color'))
```

### 执行扫描

```python
points = lidar.scan(env)
print(f"获取 {len(points)} 个点")
```

### 移动 LiDAR

```python
new_pos = lidar.position + np.array([dx, dy, dz])
if env.is_safe(new_pos):
    lidar.position = new_pos
```

### 点云着色

```python
# 按高度着色
colors = np.zeros_like(points)
norm_z = np.clip(points[:, 2] / 5.0, 0, 1)
colors[:, 0] = norm_z        # R
colors[:, 1] = 1.0 - norm_z  # G
colors[:, 2] = 0.5           # B
```

---

## 快捷键

| 按键 | 功能 | 方向 |
|------|------|------|
| W | 前进 | X+ |
| S | 后退 | X- |
| A | 左移 | Y+ |
| D | 右移 | Y- |
| Q | 上升 | Z+ |
| E | 下降 | Z- |

---

## 故障排除

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ModuleNotFoundError: No module named 'numpy'` | 依赖未安装 | `pip install numpy` |
| `ModuleNotFoundError: No module named 'sklearn'` | sklearn 未安装 | `pip install scikit-learn` |
| `ModuleNotFoundError: No module named 'open3d'` | Open3D 未安装 | `pip install open3d` |
| Matplotlib 窗口空白 | 后端问题 | 尝试 `matplotlib.use('TkAgg')` |
| Open3D 报错 | OpenGL 不支持 | 更新显卡驱动 |
| 程序卡顿 | 分辨率过高 | 降低 res_v 和 res_h |

### 调试代码

```python
# 检查射线方向
print(f"射线数量: {len(lidar.ray_dirs)}")
print(f"第一条射线方向: {lidar.ray_dirs[0]}")

# 检查碰撞检测
t = env.check_collision(lidar.position, lidar.ray_dirs[0])
print(f"碰撞距离: {t}")

# 检查点云统计
points = lidar.scan(env)
if len(points) > 0:
    print(f"点云范围: X[{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
    print(f"          Y[{points[:,1].min():.1f}, {points[:,1].max():.1f}]")
    print(f"          Z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
```

---

## 性能优化

### 减少射线数量

```python
# 降低分辨率
self.res_v = 16  # 原 32
self.res_h = 40  # 原 80
```

### 使用 GPU 加速

```python
# Open3D 版本自动使用 GPU
# 确保安装了支持 CUDA 的 Open3D
pip install open3d-cuda
```

### 批量处理

```python
# 避免循环，使用向量化操作
# 原代码
for d in ray_dirs:
    t = check_collision(origin, d)

# 优化后
t_all = check_collision_batch(origin, ray_dirs)
```

---

## 文件结构

```
lidar_demo/
├── lidar_simulation.py              # 基础版
├── lidar_simulation_interactive.py  # 交互版
├── lidar_simulation_open3d.py       # Open3D版
├── lidar_simulation_dual.py         # 双窗口版
├── lidar_simulation_stable.py       # 稳定版
├── environment.yml                  # Conda 环境
└── docs/
    ├── textbook/                    # 文档
    └── images/                      # 图片
```

---

## 常用命令

```bash
# 安装依赖
pip install numpy matplotlib scikit-learn open3d

# 运行各版本
python lidar_simulation.py              # 基础版
python lidar_simulation_interactive.py  # 交互版
python lidar_simulation_open3d.py       # Open3D版

# 性能分析
python -m cProfile lidar_simulation.py

# 查看帮助
python lidar_simulation.py --help
```

---

*本快速参考手册配套项目：LiDAR 点云仿真演示系统*

*最后更新：2026年2月*
