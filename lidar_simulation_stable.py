import open3d as o3d
import numpy as np
import time

# --- Simulation Logic (Reused) ---

class Box:
    def __init__(self, center, size, color):
        self.center = np.array(center)
        self.size = np.array(size)
        self.color = np.array(color)
        self.min_bound = self.center - self.size / 2
        self.max_bound = self.center + self.size / 2
        
    def contains(self, point, margin=0.0):
        return np.all(point >= (self.min_bound - margin)) and np.all(point <= (self.max_bound + margin))
    
    def get_mesh(self):
        # Create an Open3D mesh for visualization
        mesh = o3d.geometry.TriangleMesh.create_box(width=self.size[0], height=self.size[1], depth=self.size[2])
        mesh.translate(self.min_bound)
        mesh.paint_uniform_color(self.color)
        mesh.compute_vertex_normals()
        return mesh

class Environment3D:
    def __init__(self):
        self.obstacles = []
        # Ground (Gray)
        self.obstacles.append(Box([0, 0, -1], [100, 100, 2], [0.5, 0.5, 0.5]))
        # Red Box
        self.obstacles.append(Box([10, 5, 2], [2, 2, 4], [1.0, 0.0, 0.0]))
        # Green Box
        self.obstacles.append(Box([15, -5, 3], [3, 6, 6], [0.0, 1.0, 0.0]))
        # Orange Box
        self.obstacles.append(Box([20, 0, 1], [1, 1, 2], [1.0, 0.5, 0.0]))
        # Wall
        self.obstacles.append(Box([40, 0, 10], [2, 40, 20], [0.7, 0.7, 0.7]))

    def is_safe(self, position, margin=0.5):
        for obs in self.obstacles:
            if obs.contains(position, margin):
                return False
        return True
        
    def get_geometries(self):
        geoms = []
        for obs in self.obstacles:
            geoms.append(obs.get_mesh())
        return geoms

class Lidar3DOptimized:
    def __init__(self, position, env_meshes):
        self.position = np.array(position, dtype=float)
        self.range = 50.0
        self.res_v = 64
        self.res_h = 200 
        
        # Create RaycastingScene
        self.scene = o3d.t.geometry.RaycastingScene()
        for mesh in env_meshes:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.scene.add_triangles(t_mesh)
            
        self.ray_dirs = self._generate_ray_dirs_tensor()

    def _generate_ray_dirs_tensor(self):
        v_angles = np.linspace(np.radians(-15), np.radians(15), self.res_v)
        h_angles = np.linspace(np.radians(0), np.radians(360), self.res_h)
        v, h = np.meshgrid(v_angles, h_angles)
        dx = np.cos(v) * np.cos(h)
        dy = np.cos(v) * np.sin(h)
        dz = np.sin(v)
        dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        return o3d.core.Tensor(dirs, dtype=o3d.core.float32)

    def scan(self, position):
        N = self.ray_dirs.shape[0]
        origins = np.tile(position, (N, 1))
        # Reduce memory usage by not creating massive python arrays if possible
        rays_np = np.concatenate([origins, self.ray_dirs.numpy()], axis=1)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.float32)
        
        result = self.scene.cast_rays(rays)
        t_hit = result['t_hit'].numpy()
        valid_mask = (t_hit < self.range) & (t_hit > 0)
        
        valid_t = t_hit[valid_mask]
        valid_dirs = self.ray_dirs.numpy()[valid_mask]
        points = origins[valid_mask] + valid_dirs * valid_t[:, np.newaxis]
        
        return points

# --- Single Window Stable Implementation ---

class LidarApp:
    def __init__(self):
        self.env = Environment3D()
        self.lidar_pos = np.array([0.0, 0.0, 2.0])
        self.lidar = Lidar3DOptimized(self.lidar_pos, self.env.get_geometries())
        
        # Single Window for everything
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="LiDAR Simulation (Single View)", width=1280, height=720)
        
        # Setup Render Options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 3.0
        
        # Add Environment Meshes
        self.env_meshes = self.env.get_geometries()
        for mesh in self.env_meshes:
            mesh.compute_vertex_normals()
            self.vis.add_geometry(mesh)
            
        # Lidar Marker (Red Sphere)
        self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
        self.lidar_sphere.paint_uniform_color([1, 0, 0])
        self.lidar_sphere.compute_vertex_normals()
        self.lidar_sphere.translate(self.lidar_pos)
        self.vis.add_geometry(self.lidar_sphere)
        
        # Point Cloud (Initially Empty)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # Controls
        self.vis.register_key_callback(ord('W'), self.move_fwd)
        self.vis.register_key_callback(ord('S'), self.move_bwd)
        self.vis.register_key_callback(ord('A'), self.move_left)
        self.vis.register_key_callback(ord('D'), self.move_right)
        self.vis.register_key_callback(ord('Q'), self.move_up)
        self.vis.register_key_callback(ord('E'), self.move_down)

        print("Controls: WASD (Move), Q/E (Up/Down)")
        self.run()
        
    def move(self, delta):
        new_pos = self.lidar_pos + delta
        if self.env.is_safe(new_pos):
            self.lidar_pos = new_pos

    # Callback signatures: (vis)
    def move_fwd(self, vis): self.move(np.array([0.5, 0, 0]))
    def move_bwd(self, vis): self.move(np.array([-0.5, 0, 0]))
    def move_left(self, vis): self.move(np.array([0, 0.5, 0]))
    def move_right(self, vis): self.move(np.array([0, -0.5, 0]))
    def move_up(self, vis): self.move(np.array([0, 0, 0.5]))
    def move_down(self, vis): self.move(np.array([0, 0, -0.5]))

    def run(self):
        while True:
            # 1. Update Lidar Geometry
            # Simplest Open3D legacy update: Remove old, Add new instance
            self.vis.remove_geometry(self.lidar_sphere, reset_bounding_box=False)
            
            self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
            self.lidar_sphere.paint_uniform_color([1, 0, 0])
            self.lidar_sphere.compute_vertex_normals()
            self.lidar_sphere.translate(self.lidar_pos)
            
            self.vis.add_geometry(self.lidar_sphere, reset_bounding_box=False)
            
            # 2. Update Scan
            points = self.lidar.scan(self.lidar_pos)
            
            if len(points) > 0:
                self.pcd.points = o3d.utility.Vector3dVector(points)
                
                # Color Points: Gradient by Height
                colors = np.zeros_like(points)
                norm_z = np.clip(points[:, 2] / 5.0, 0, 1)
                colors[:, 0] = norm_z          # R
                colors[:, 1] = 1.0 - norm_z    # G
                colors[:, 2] = 1.0             # B
                
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.vis.update_geometry(self.pcd)
            else:
                self.pcd.points = o3d.utility.Vector3dVector([])
                self.vis.update_geometry(self.pcd)
            
            # 3. Render
            if not self.vis.poll_events():
                break
            
            self.vis.update_renderer()
            
            # Cap FPS
            time.sleep(0.01)
            
        self.vis.destroy_window()

if __name__ == "__main__":
    LidarApp()
