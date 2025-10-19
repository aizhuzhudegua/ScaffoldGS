import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays, getProjectionMatrixCenterShift  # 新增getProjectionMatrixCenterShift


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, fx=None, fy=None, cx=None, cy=None,  # 新增cx/cy/fx/fy参数
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 height=None, width=None, depth=None, normal=None, image_mask=None):  # 新增可选的深度/法线/掩码参数
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        # 新增：存储cx/cy/fx/fy内参
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # 处理图像与分辨率（兼容原有逻辑，新增无图像时用width/height）
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            # 应用gt_alpha_mask（原有逻辑）
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            # 无图像时，用传入的width/height作为分辨率
            self.original_image = None
            self.image_width = width
            self.image_height = height

        # 新增：处理深度、法线、图像掩码（对齐目标代码）
        if depth is not None:
            self.depth = depth.to(self.data_device)
        else:
            self.depth = torch.zeros((1, self.image_height, self.image_width), dtype=torch.float32, device=self.data_device)

        if normal is not None:
            self.normal = normal.to(self.data_device)
        else:
            self.normal = torch.zeros((3, self.image_height, self.image_width), dtype=torch.float32, device=self.data_device)

        if image_mask is not None:
            self.image_mask = image_mask.to(self.data_device)
        else:
            self.image_mask = torch.ones_like(self.depth)

        # 原有：近远裁剪面
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        # 原有：视图变换矩阵（世界→相机）
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        
        # 新增：根据是否有fx，选择不同的投影矩阵计算方式（对齐目标代码）
        if self.fx is None:
            # 无内参时，用视场角计算投影矩阵（原有逻辑）
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        else:
            # 有内参时，用cx/cy/fx/fy计算带中心偏移的投影矩阵
            self.projection_matrix = getProjectionMatrixCenterShift(
                self.znear, self.zfar, self.cx, self.cy, self.fx, self.fy, self.image_width, self.image_height
            ).transpose(0, 1).to(self.data_device)

        # 原有：完整投影变换（视图→投影）
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 原有：相机中心（世界坐标系）
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # 新增：对齐目标代码的关键属性
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()  # 相机→世界变换矩阵
        self.intrinsics = self.get_intrinsics()  # 内参矩阵（新增方法）
        self.extrinsics = self.get_extrinsics()  # 外参矩阵（新增方法）
        self.proj_matrix = self.get_proj_matrix()  # 内参×外参矩阵（新增方法）

    # 新增：获取内参矩阵（与目标代码完全一致）
    def get_intrinsics(self):
        if self.fx is None:
            # 无内参时，用视场角和分辨率计算（原有逻辑）
            focal_x = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            focal_y = self.image_height / (2 * np.tan(self.FoVy * 0.5))
            return torch.tensor([[focal_x, 0, self.image_width / 2],
                                 [0, focal_y, self.image_height / 2],
                                 [0, 0, 1]], device=self.data_device, dtype=torch.float32)
        else:
            # 有内参时，直接用cx/cy/fx/fy生成
            return torch.tensor([[self.fx, 0, self.cx],
                                 [0, self.fy, self.cy],
                                 [0, 0, 1]], device=self.data_device, dtype=torch.float32)

    # 新增：获取外参矩阵（相机→世界，与目标代码一致）
    def get_extrinsics(self):
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R.transpose()  # 外参R为世界→相机，转置后为相机→世界
        Rt[:3, 3] = self.T
        return torch.from_numpy(Rt).float().to(self.data_device)

    # 新增：获取内参×外参矩阵（与目标代码一致）
    def get_proj_matrix(self):
        eK_mat = torch.eye(4, dtype=self.intrinsics.dtype, device=self.intrinsics.device)
        eK_mat[0:3, 0:3] = self.intrinsics  # 扩展内参为4x4矩阵
        return torch.bmm(eK_mat.unsqueeze(0), self.extrinsics.unsqueeze(0)).squeeze(0)

    # 新增：获取相机主坐标轴（与目标代码一致）
    def get_primary_axis(self):
        p_axis = torch.zeros([3], dtype=torch.float32, device=self.data_device)
        p_axis[2] = 1  # 相机局部z轴（朝前）
        p_axis_world = self.c2w[:3, :3] @ p_axis  # 转换到世界坐标系
        return p_axis_world

    # 新增：获取世界坐标系下的光线方向（修复cx/cy偏移问题，与目标代码一致）
    def get_world_directions(self):
        # 生成像素网格（indexing="ij"确保v对应行、u对应列）
        v, u = torch.meshgrid(
            torch.arange(self.image_height, device=self.data_device),
            torch.arange(self.image_width, device=self.data_device),
            indexing="ij"
        )
        # 用cx/cy校正光线方向（避免中心偏移）
        focal_x = self.intrinsics[0, 0]
        focal_y = self.intrinsics[1, 1]
        directions = torch.stack([
            (u - self.intrinsics[0, 2]) / focal_x,  # x方向：(像素u - 中心cx)/焦距fx
            (v - self.intrinsics[1, 2]) / focal_y,  # y方向：(像素v - 中心cy)/焦距fy
            torch.ones_like(u)  # z方向：1（相机局部坐标系朝前）
        ], dim=0)
        # 归一化光线方向
        directions = F.normalize(directions, dim=0)
        # 转换到世界坐标系（相机→世界）
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    # 原有：获取NERF格式的内外参（更新为用cx/cy内参）
    def get_calib_matrix_nerf(self):
        if self.fx is None:
            # 无内参时，用视场角计算焦距（原有逻辑）
            focal = fov2focal(self.FoVx, self.image_width)
            intrinsic_matrix = torch.tensor([[focal, 0, self.image_width / 2], 
                                             [0, focal, self.image_height / 2], 
                                             [0, 0, 1]], device=self.data_device).float()
        else:
            # 有内参时，用cx/cy/fx/fy生成（新增逻辑）
            intrinsic_matrix = torch.tensor([[self.fx, 0, self.cx], 
                                             [0, self.fy, self.cy], 
                                             [0, 0, 1]], device=self.data_device).float()
        # 外参：相机→世界变换矩阵（原有逻辑）
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()
        return intrinsic_matrix, extrinsic_matrix

    # 原有：获取光线方向（更新为用get_calib_matrix_nerf的内参，支持cx/cy）
    def get_rays(self):
        intrinsic_matrix, extrinsic_matrix = self.get_calib_matrix_nerf()
        # 调用utils的get_rays，此时内参已包含cx/cy，光线方向会自动校正
        viewdirs = get_rays(self.image_width, self.image_height, intrinsic_matrix, extrinsic_matrix[:3, :3])
        return viewdirs

    # 新增：静态方法，用于GUI创建相机（与目标代码一致）
    @staticmethod
    def create_for_gui():
        return Camera(
            colmap_id=0, R=np.eye(3), T=np.zeros(3), 
            FoVx=50, FoVy=50, image=None, gt_alpha_mask=None,
            image_name="gui", uid="gui",
            fx=None, fy=None, cx=None, cy=None,  # GUI默认无内参
            width=800, height=600  # GUI默认分辨率
        )


# 原有：MiniCam类（保留，无修改）
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]