from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces

import numpy as np
import torch
from pathlib import Path

import numpy as np
import open3d as o3d
import imageio

import numpy as np
import torch
import open3d as o3d
import imageio

def color_parts(source):
    """
    source: [N,P,6] numpy / tensor / [P,6]
    返回合并后的带颜色的 open3d PointCloud
    """
    if torch.is_tensor(source):
        arr = source.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(source, dtype=np.float32)

    if arr.ndim == 2:  # [P,6] → [1,P,6]
        arr = arr[None, ...]
    N, P, D = arr.shape

    rng = np.random.default_rng(42)
    colors = rng.random((N, 3))  # 每个 part 一种随机颜色

    all_points, all_colors = [], []
    for i in range(N):
        xyz = arr[i, :, :3]
        col = np.tile(colors[i], (xyz.shape[0], 1))
        all_points.append(xyz)
        all_colors.append(col)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    return pcd

def render_parts_to_gif(source, gif_path="rotation.gif", n_frames=60, width=800, height=800):
    pcd = color_parts(source)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    images = []
    for i in range(n_frames):
        # 每一帧旋转 360/n_frames 度 (绕 Y 轴)
        R = pcd.get_rotation_matrix_from_axis_angle([0, 2*np.pi/n_frames, 0])
        pcd.rotate(R, center=(0,0,0))

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        img = (255 * np.asarray(img)).astype(np.uint8)
        images.append(img)

    vis.destroy_window()
    imageio.mimsave(gif_path, images, fps=15)



def render_ply_to_gif(ply_path, gif_path="rotation.gif", n_frames=60, width=800, height=800):
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    images = []
    for i in range(n_frames):
        # 每一帧旋转 360/n_frames 度
        R = pcd.get_rotation_matrix_from_axis_angle([0, 2*np.pi/n_frames, 0])  # 绕y轴旋转
        pcd.rotate(R, center=(0,0,0))
        
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 截图
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (255 * np.asarray(img)).astype(np.uint8)
        images.append(img)

    vis.destroy_window()

    # 保存为 gif
    imageio.mimsave(gif_path, images, fps=15)

def save_all_parts_to_colored_ply(source, ply_path, num_samples=1024):
    """
    source: np.ndarray / torch.Tensor / str (path to .npy)
            shape: [N, P, 6] or [P, 6]
            where last dim = [x,y,z,(nx,ny,nz)]
    ply_path: output PLY file
    num_samples: 每个 part 采样的点数
    """

    # Load
    if isinstance(source, (str, Path)):
        npy = np.load(source, allow_pickle=True).item()
        parts = npy.get("parts", [])
        parts = parts if (isinstance(parts, (list, tuple)) and len(parts) > 0) else [npy["object"]]
        arr = np.asarray(parts, dtype=np.float32)
    elif torch.is_tensor(source):
        arr = source.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(source, dtype=np.float32)

    # Normalize shape
    if arr.ndim == 2:  # [P,6] → [1,P,6]
        arr = arr[None, ...]
    N, P, D = arr.shape

    rng = np.random.default_rng(42)  # 固定随机种子，保证可复现

    # Write PLY header
    total_vertices = N * num_samples
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 给每个 part 随机一个颜色
        colors = rng.integers(0, 255, size=(N, 3))

        for i in range(N):
            xyz = arr[i, :, :3]
            # 随机采样
            if xyz.shape[0] > num_samples:
                idx = rng.choice(xyz.shape[0], size=num_samples, replace=False)
                xyz = xyz[idx]
            elif xyz.shape[0] < num_samples:
                idx = rng.choice(xyz.shape[0], size=num_samples, replace=True)
                xyz = xyz[idx]
            # 写入
            color = colors[i]
            for p in xyz:
                f.write(f"{p[0]} {p[1]} {p[2]} {color[0]} {color[1]} {color[2]}\n")


class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
        image_path = data_config['image_path']
        image = Image.open(image_path).resize(self.image_size)
        if random.random() < self.rotating_ratio:
            image = self.transform(image)
        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
        images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]

        save_all_parts_to_colored_ply(part_surfaces, 'demo2.ply', num_samples=20000)
        img = images[0].cpu().numpy().astype(np.uint8)
        imageio.imwrite("demo2_image.png", img)
        # render_parts_to_gif(part_surfaces, gif_path="colored_parts.gif", n_frames=60)
        return {
            "images": images,
            "part_surfaces": part_surfaces,
        }
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                # Successfully get a batch
                if len(temp_batch) < batch_size:
                    # pad the batch
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
                # Else, the code enters here because len(data_configs) == 0
                # which means in the left data_configs, there are no enough 
                # "suitable" configs to form a batch. 
                # Thus, drop the uncompleted batch.
        progress_bar.close()
        return batched_data_configs
        
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # placeholder
            return {}
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
        }
        return batch