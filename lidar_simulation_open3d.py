import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import time
import threading

# --- Shared Logic ---

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
        # Reduce memory usage by not creating massive python arrays if possible, but python fallback is fine for < 1M points
        rays_np = np.concatenate([origins, self.ray_dirs.numpy()], axis=1)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.float32)
        
        result = self.scene.cast_rays(rays)
        t_hit = result['t_hit'].numpy()
        valid_mask = (t_hit < self.range) & (t_hit > 0)
        
        valid_t = t_hit[valid_mask]
        valid_dirs = self.ray_dirs.numpy()[valid_mask]
        points = origins[valid_mask] + valid_dirs * valid_t[:, np.newaxis]
        
        return points

# --- GUI Application ---

class LidarApp:
    def __init__(self):
        self.env = Environment3D()
        self.lidar_pos = np.array([0.0, 0.0, 2.0])
        self.lidar = Lidar3DOptimized(self.lidar_pos, self.env.get_geometries())
        self.running = True
        
        # GUI Setup
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window("Open3D LiDAR Simulation (Split View)", 1600, 800)
        self.window.set_on_close(self.on_close)
        
        # UI Layout: Horizontal split
        self.layout = gui.Horiz()
        self.window.add_child(self.layout)
        
        # Left Scene: Simulation World
        self.scene_sim = gui.SceneWidget()
        self.scene_sim.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_sim.scene.set_background([0.9, 0.9, 0.9, 1.0])
        self.layout.add_child(self.scene_sim)
        self.layout.add_fixed(2) # Divider
        
        # Right Scene: Point Cloud
        self.scene_pc = gui.SceneWidget()
        self.scene_pc.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_pc.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self.layout.add_child(self.scene_pc)
        
        # Setup Geometries
        # 1. World Scene
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        for i, mesh in enumerate(self.env.get_geometries()):
            self.scene_sim.scene.add_geometry(f"obs_{i}", mesh, mat)
            
        # Lidar Marker (Sphere)
        self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
        self.lidar_sphere.paint_uniform_color([0, 0, 0])
        self.lidar_sphere.compute_vertex_normals()
        self.scene_sim.scene.add_geometry("lidar", self.lidar_sphere, mat)
        
        # 2. Point Cloud Scene
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = 3.0
        
        # Initial empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        self.scene_pc.scene.add_geometry("pcd", self.pcd, self.pcd_mat)
        
        # Camera Setup
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, 0], [50, 50, 10])
        self.scene_sim.setup_camera(60, bbox, [0, 0, 0])
        self.scene_pc.setup_camera(60, bbox, [0, 0, 0])
        
        # Input State
        self.keys = set()
        self.scene_sim.set_on_key(self.on_key)
        self.scene_pc.set_on_key(self.on_key) # Handle keys on both sides
        
    def on_close(self):
        self.running = False
        return True
        
    def on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            self.keys.add(event.key)
        elif event.type == gui.KeyEvent.UP:
            if event.key in self.keys:
                self.keys.remove(event.key)
        return True # Handled

    def update_render(self, points, new_pos):
        # This function runs on the Main Thread
        
        # 1. Update Lidar in Sim View
        self.scene_sim.scene.remove_geometry("lidar") # This is fast enough for simple geom
        self.lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(0.2)
        self.lidar_sphere.translate(new_pos)
        self.lidar_sphere.paint_uniform_color([0, 0, 0])
        self.scene_sim.scene.add_geometry("lidar", self.lidar_sphere, rendering.MaterialRecord())
        
        # 2. Update Point Cloud
        if len(points) > 0:
            self.scene_pc.scene.remove_geometry("pcd")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            colors = np.zeros_like(points)
            if len(points) > 0:
                norm_z = np.clip(points[:, 2] / 5.0, 0, 1)
                colors[:, 0] = norm_z
                colors[:, 1] = 1.0 - norm_z
                colors[:, 2] = 0.5
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.scene_pc.scene.add_geometry("pcd", pcd, self.pcd_mat)
            
        self.window.post_redraw()

    def run_simulation_thread(self):
        # Wait a bit for window to actually appear
        time.sleep(1.0)
        
        while self.running:
            # --- Logic (Background Thread) ---
            
            # 1. Handle Movement
            step = 0.5
            delta = np.zeros(3)
            
            # Read keys (Thread-safe enough for this)
            if gui.KeyName.W in self.keys or 119 in self.keys or 87 in self.keys: delta[0] += step
            if gui.KeyName.S in self.keys or 115 in self.keys or 83 in self.keys: delta[0] -= step
            if gui.KeyName.A in self.keys or 97 in self.keys or 65 in self.keys: delta[1] += step
            if gui.KeyName.D in self.keys or 100 in self.keys or 68 in self.keys: delta[1] -= step
            if gui.KeyName.Q in self.keys or 113 in self.keys or 81 in self.keys: delta[2] += step
            if gui.KeyName.E in self.keys or 101 in self.keys or 69 in self.keys: delta[2] -= step
            
            if np.any(delta):
                new_pos = self.lidar_pos + delta
                if self.env.is_safe(new_pos):
                    self.lidar_pos = new_pos
            
            # 2. Raycast (Heavy)
            points = self.lidar.scan(self.lidar_pos)
            
            # --- Render Request (Main Thread) ---
            # Pass copies of data to avoid race conditions during render
            self.app.post_to_main_thread(self.window, lambda: self.update_render(points, self.lidar_pos))
            
            # Control framerate of simulation
            time.sleep(0.01) # ~100 Hz simulation rate

    def run(self):
        # Start Thread HERE
        threading.Thread(target=self.run_simulation_thread, daemon=True).start()
        self.app.run()

if __name__ == "__main__":
    LidarApp().run()
