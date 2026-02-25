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
        # Reduce memory usage
        rays_np = np.concatenate([origins, self.ray_dirs.numpy()], axis=1)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.float32)
        
        result = self.scene.cast_rays(rays)
        t_hit = result['t_hit'].numpy()
        valid_mask = (t_hit < self.range) & (t_hit > 0)
        
        valid_t = t_hit[valid_mask]
        valid_dirs = self.ray_dirs.numpy()[valid_mask]
        points = origins[valid_mask] + valid_dirs * valid_t[:, np.newaxis]
        
        return points

# --- Dual Window Stable Implementation (Legacy API) ---

class LidarAppDual:
    def __init__(self):
        self.env = Environment3D()
        self.lidar_pos = np.array([0.0, 0.0, 2.0])
        self.lidar = Lidar3DOptimized(self.lidar_pos, self.env.get_geometries())
        
        # Window 1: Simulation World
        self.vis_sim = o3d.visualization.VisualizerWithKeyCallback()
        self.vis_sim.create_window(window_name="World Simulation (Control Here)", width=800, height=600, left=0, top=0)
        
        # Window 2: Point Cloud
        self.vis_pc = o3d.visualization.Visualizer()
        self.vis_pc.create_window(window_name="Point Cloud View", width=800, height=600, left=820, top=0)
        
        # Setup Sim World
        self.vis_sim.get_render_option().background_color = np.array([0.9, 0.9, 0.9])
        for mesh in self.env.get_geometries():
            mesh.compute_vertex_normals()
            self.vis_sim.add_geometry(mesh)
            
        self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
        self.lidar_sphere.paint_uniform_color([0, 0, 0])
        self.lidar_sphere.compute_vertex_normals()
        self.lidar_sphere.translate(self.lidar_pos)
        self.vis_sim.add_geometry(self.lidar_sphere)
        
        # Setup Point Cloud in PC Window
        self.vis_pc.get_render_option().background_color = np.array([0.05, 0.05, 0.05]) # Dark Grey
        self.vis_pc.get_render_option().point_size = 3.0
        
        # We need two separate geometry objects for two windows usually to avoid context issues, 
        # or just add the same one. Open3D geometries are data pointers, visuals are internal. 
        # But for remove/add, we need to manage them.
        self.pcd_sim = o3d.geometry.PointCloud()
        self.pcd_pc = o3d.geometry.PointCloud()
        
        self.vis_sim.add_geometry(self.pcd_sim)
        self.vis_pc.add_geometry(self.pcd_pc)
        
        # Initialize Cameras
        self.vis_sim.poll_events()
        self.vis_sim.update_renderer()
        self.vis_pc.poll_events()
        self.vis_pc.update_renderer()
        
        # Controls (Register on Sim Window mostly)
        self.vis_sim.register_key_callback(ord('W'), self.move_fwd)
        self.vis_sim.register_key_callback(ord('S'), self.move_bwd)
        self.vis_sim.register_key_callback(ord('A'), self.move_left)
        self.vis_sim.register_key_callback(ord('D'), self.move_right)
        self.vis_sim.register_key_callback(ord('Q'), self.move_up)
        self.vis_sim.register_key_callback(ord('E'), self.move_down)

        print("Controls: Focus 'World Simulation' window and use WASD + QE")
        self.run()
        
    def move(self, delta):
        new_pos = self.lidar_pos + delta
        if self.env.is_safe(new_pos):
            self.lidar_pos = new_pos

    def move_fwd(self, vis): self.move(np.array([0.5, 0, 0]))
    def move_bwd(self, vis): self.move(np.array([-0.5, 0, 0]))
    def move_left(self, vis): self.move(np.array([0, 0.5, 0]))
    def move_right(self, vis): self.move(np.array([0, -0.5, 0]))
    def move_up(self, vis): self.move(np.array([0, 0, 0.5]))
    def move_down(self, vis): self.move(np.array([0, 0, -0.5]))

    def run(self):
        first_frame = True
        while True:
            # 1. Update Sim View: Lidar Sphere
            self.vis_sim.remove_geometry(self.lidar_sphere, reset_bounding_box=False)
            self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
            self.lidar_sphere.paint_uniform_color([0, 0, 0])
            self.lidar_sphere.compute_vertex_normals()
            self.lidar_sphere.translate(self.lidar_pos)
            self.vis_sim.add_geometry(self.lidar_sphere, reset_bounding_box=False)
            
            # 2. Update Scan
            points = self.lidar.scan(self.lidar_pos)
            
            if len(points) > 0:
                # Prepare colors
                colors = np.zeros_like(points)
                norm_z = np.clip(points[:, 2] / 5.0, 0, 1)
                colors[:, 0] = norm_z
                colors[:, 1] = 1.0 - norm_z
                colors[:, 2] = 0.5
                
                # Update PCD in Sim Window
                self.vis_sim.remove_geometry(self.pcd_sim, reset_bounding_box=False)
                self.pcd_sim = o3d.geometry.PointCloud()
                self.pcd_sim.points = o3d.utility.Vector3dVector(points)
                self.pcd_sim.colors = o3d.utility.Vector3dVector(colors)
                self.vis_sim.add_geometry(self.pcd_sim, reset_bounding_box=False)
                
                # Update PCD in PC Window
                self.vis_pc.remove_geometry(self.pcd_pc, reset_bounding_box=False)
                self.pcd_pc = o3d.geometry.PointCloud()
                self.pcd_pc.points = o3d.utility.Vector3dVector(points)
                self.pcd_pc.colors = o3d.utility.Vector3dVector(colors)
                self.vis_pc.add_geometry(self.pcd_pc, reset_bounding_box=first_frame) # Reset view only on first valid frame? 
                
                if first_frame:
                    self.vis_pc.reset_view_point(True)
                    first_frame = False
            else:
                 # Empty... just do nothing or clear
                 pass
            
            # 3. Update Renderers
            if not self.vis_sim.poll_events(): break
            self.vis_sim.update_renderer()
            
            if not self.vis_pc.poll_events(): break
            self.vis_pc.update_renderer()
            
            time.sleep(0.01)
            
        self.vis_sim.destroy_window()
        self.vis_pc.destroy_window()

if __name__ == "__main__":
    LidarAppDual()
