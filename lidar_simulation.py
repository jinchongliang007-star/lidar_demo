import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

# Attempt to import sklearn for clustering
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not found. Clustering will be disabled.")

class Box:
    def __init__(self, center, size, color='blue'):
        self.center = np.array(center)
        self.size = np.array(size)
        self.color = color
        self.min_bound = self.center - self.size / 2
        self.max_bound = self.center + self.size / 2
        
    def contains(self, point, margin=0.0):
        # Check if point is inside box with some margin
        return np.all(point >= (self.min_bound - margin)) and np.all(point <= (self.max_bound + margin))

    def intersect(self, ray_origin, ray_dir):
        # Slab method for AABB intersection
        # ray_dir should be normalized
        
        # Avoid division by zero
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
        # Ground
        self.obstacles.append(Box([0, 0, -1], [100, 100, 2], color='gray')) 
        
        # Obstacles
        self.obstacles.append(Box([10, 5, 2], [2, 2, 4], color='red'))
        self.obstacles.append(Box([15, -5, 3], [3, 6, 6], color='green'))
        self.obstacles.append(Box([20, 0, 1], [1, 1, 2], color='orange'))
        
        # Wall
        self.obstacles.append(Box([40, 0, 10], [2, 40, 20], color='gray'))

    def check_collision(self, ray_origin, ray_dir):
        closest_t = np.inf
        
        for obs in self.obstacles:
            t = obs.intersect(ray_origin, ray_dir)
            if t < closest_t:
                closest_t = t
                
        return closest_t

    def is_safe(self, position, margin=0.5):
        # Check if position is inside any obstacle
        for obs in self.obstacles:
            if obs.contains(position, margin):
                return False
        return True

class Lidar3D:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.range = 50.0
        self.fov_v = (-15, 15) # Degrees
        self.fov_h = (0, 360) # Degrees
        self.res_v = 15 # Number of beams vertical
        self.res_h = 60 # Number of beams horizontal
        
        # Precompute ray directions
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
            # Add noise to direction
            noisy_d = d + np.random.normal(0, 0.005, 3)
            noisy_d = noisy_d / np.linalg.norm(noisy_d)
            
            t = env.check_collision(self.position, noisy_d)
            
            if t < self.range:
                # Add measurement noise
                noisy_t = t + np.random.normal(0, 0.05)
                if noisy_t > 0:
                    point = self.position + noisy_d * noisy_t
                    points.append(point)
                    
        return np.array(points)

def cluster_points(points):
    if not HAS_SKLEARN or len(points) == 0:
        return np.zeros(len(points)) # All same cluster
        
    clustering = DBSCAN(eps=2.0, min_samples=3).fit(points)
    return clustering.labels_

def draw_box(ax, box):
    # Draw wireframe box
    # 8 corners
    c = box.center
    s = box.size / 2
    
    # Create all combinations of corners
    x = [c[0]-s[0], c[0]+s[0]]
    y = [c[1]-s[1], c[1]+s[1]]
    z = [c[2]-s[2], c[2]+s[2]]
    
    # Simple line drawing
    # Bottom rect
    ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], [z[0], z[0], z[0], z[0], z[0]], color=box.color, alpha=0.5)
    # Top rect
    ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], [z[1], z[1], z[1], z[1], z[1]], color=box.color, alpha=0.5)
    # Vertical lines
    ax.plot([x[0], x[0]], [y[0], y[0]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[1], x[1]], [y[0], y[0]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[1], x[1]], [y[1], y[1]], [z[0], z[1]], color=box.color, alpha=0.5)
    ax.plot([x[0], x[0]], [y[1], y[1]], [z[0], z[1]], color=box.color, alpha=0.5)

def update(frame, lidar, env, ax_sim, ax_pc, status_text_sim, status_text_pc):
    # Move Lidar with collision check
    # Proposed movement
    new_x = lidar.position[0] + 0.2
    new_y = 5 * np.sin(frame * 0.1)
    new_z = lidar.position[2]
    
    proposed_pos = np.array([new_x, new_y, new_z])
    
    if env.is_safe(proposed_pos):
        lidar.position = proposed_pos
    else:
        # Simple collision handling: Stop X movement, or bounce? 
        # For this demo, let's just slide along Y or stop
        # Let's try only moving Y
        proposed_pos_y = np.array([lidar.position[0], new_y, new_z])
        if env.is_safe(proposed_pos_y):
             lidar.position = proposed_pos_y
        else:
             # Stuck? Just stay
             pass
    
    # Scan
    points = lidar.scan(env)
    
    # --- Update Simulation View ---
    # Clear collections but keep environment lines?
    # Clearing completely is easiest but slow.
    # To properly clear:
    for collection in list(ax_sim.collections):
        collection.remove()
    # Also clear scatter points if any
        
    # Draw Lidar Position
    ax_sim.scatter([lidar.position[0]], [lidar.position[1]], [lidar.position[2]], color='black', s=50, marker='^', label='Lidar')
    
    # --- Update Point Cloud View ---
    for collection in list(ax_pc.collections):
        collection.remove()
        
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
                
        # Plot on Point Cloud View
        ax_pc.scatter(points[:, 0], points[:, 1], points[:, 2], c=c_array, s=5)
        
        # Also plot the sensing rays on Sim View (optional, maybe heavy? Let's just plot impact points for context)
        # ax_sim.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.3)
        
        status_text_pc.set_text(f"Clusters: {len(unique_labels)} | Points: {len(points)}")
    else:
        status_text_pc.set_text(f"Clusters: 0 | Points: 0")
        
    status_text_sim.set_text(f"Lidar Pos: {lidar.position[:2].round(1)}")
        
    # Tracking
    for ax in [ax_sim, ax_pc]:
        ax.set_xlim(lidar.position[0] - 20, lidar.position[0] + 20)
        ax.set_ylim(-20, 20)
        
    return status_text_sim, status_text_pc

def run_simulation():
    env = Environment3D()
    lidar = Lidar3D([0, 0, 2])
    
    # Setup Dual Plot
    fig = plt.figure(figsize=(14, 7))
    
    # Subplot 1: Simulation (Environment + Lidar)
    ax_sim = fig.add_subplot(121, projection='3d')
    ax_sim.set_title("Simulation Environment (True World)")
    ax_sim.set_xlabel("X")
    ax_sim.set_ylabel("Y")
    ax_sim.set_zlabel("Z")
    ax_sim.set_zlim(0, 10)
    
    # Draw Environment (Static)
    for obs in env.obstacles:
        draw_box(ax_sim, obs)
        
    # Subplot 2: Point Cloud
    ax_pc = fig.add_subplot(122, projection='3d')
    ax_pc.set_title("Lidar Point Cloud (Sensor Data)")
    ax_pc.set_xlabel("X")
    ax_pc.set_ylabel("Y")
    ax_pc.set_zlabel("Z")
    ax_pc.set_zlim(0, 10)
    
    status_text_sim = ax_sim.text2D(0.05, 0.95, "", transform=ax_sim.transAxes)
    status_text_pc = ax_pc.text2D(0.05, 0.95, "", transform=ax_pc.transAxes)
    
    ani = animation.FuncAnimation(fig, update, fargs=(lidar, env, ax_sim, ax_pc, status_text_sim, status_text_pc), 
                                  frames=200, interval=100, blit=False)
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
