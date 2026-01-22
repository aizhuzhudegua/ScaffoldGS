#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map
from .r3dg_rasterization import (
    GaussianRasterizationSettings as RasterSettings,
    GaussianRasterizer as Rasterizer
)
import torch.nn.functional as F

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    n_visible_anchors = anchor.shape[0]  # 可见锚点数量

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c] n个锚点 c个特征维度

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3] without dist
    
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
   
    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color (now as diffuse color)
    diffuse_color = pc.get_color_mlp(cat_local_view_wodist)
    diffuse_color = diffuse_color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    specular = pc.get_specular_mlp(cat_local_view_wodist)
    specular = specular.reshape([anchor.shape[0]*pc.n_offsets, 3])

    roughness = pc.get_roughness_mlp(cat_local_view_wodist)
    roughness = roughness.reshape([anchor.shape[0]*pc.n_offsets, 1])

    normal1 = pc.get_normal1_mlp(cat_local_view_wodist)
    normal1 = normal1.reshape([anchor.shape[0]*pc.n_offsets, 3])
    normal2 = pc.get_normal2_mlp(cat_local_view_wodist)
    normal2 = normal2.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    # -------------------------- 新增：生成锚点-高斯归属索引 --------------------------
    # 1. 生成可见锚点的原始索引（0 ~ n_visible_anchors-1）
    anchor_indices_raw = torch.arange(n_visible_anchors, device=anchor.device)  # [n_visible_anchors]
    # 2. 复制索引：每个锚点的索引重复k次（与下属高斯数量匹配）
    anchor_indices_repeated = repeat(anchor_indices_raw, 'n -> (n k)', k=pc.n_offsets)  # [n_visible_anchors * k]
    # -----------------------------------------------------------------------------
    
    # combine for parallel masking：新增锚点索引拼接
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # 拼接时加入复制后的锚点索引（最后一列）
    concatenated_all = torch.cat([
        concatenated_repeated, diffuse_color, scale_rot, offsets, 
        normal1, normal2, specular, roughness, 
        anchor_indices_repeated.unsqueeze(1)  # 新增：锚点索引列
    ], dim=-1)
    
    masked = concatenated_all[mask]
    # 拆分时同步提取锚点索引（调整split列数，最后1列为索引）
    scaling_repeat, repeat_anchor, diffuse_color, scale_rot, offsets, normal1, normal2, specular, roughness, anchor_indices_valid = masked.split(
        [6, 3, 3, 7, 3, 3, 3, 3, 1, 1],  # 最后一个1对应锚点索引
        dim=-1
    )
    # 索引转长整型并压缩维度
    anchor_indices_valid = anchor_indices_valid.long().squeeze(1)  # [N]，N为有效高斯数量
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    # 返回新增的锚点索引（用于后续法线平均）
    if is_training:
        return xyz, diffuse_color, opacity, scaling, rot, neural_opacity, mask, normal1, normal2, specular, roughness, anchor_indices_valid
    else:
        return xyz, diffuse_color, opacity, scaling, rot, normal1, normal2, specular, roughness, anchor_indices_valid


def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

# render 360 lighting for a single gaussian
def render_lighting(pc : GaussianModel, resolution=(512, 1024), sampled_index=None):
    if pc.brdf_mode=="envmap":
        lighting = extract_env_map(pc.brdf_mlp, resolution) # (H, W, 3)
        lighting = lighting.permute(2,0,1) # (3, H, W)
    else:
        raise NotImplementedError

    return lighting

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False , pbr = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # 是否训练了颜色mlp
    is_training = pc.get_color_mlp.training
        
    # 接收新增的锚点索引
    if is_training:
        xyz, diffuse_color, opacity, scaling, rot, neural_opacity, mask, normal1, normal2, specular, roughness, anchor_indices_valid  = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, diffuse_color, opacity, scaling, rot, normal1, normal2, specular, roughness, anchor_indices_valid = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    gb_pos = xyz # (N, 3)，N为有效高斯数量
    view_pos = viewpoint_camera.camera_center.repeat(gb_pos.shape[0], 1) # (N, 3)

    # 计算每个高斯的视角方向
    dir_pp = (gb_pos - view_pos)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)

    # 原始法线计算（带残差）
    normal, delta_normal = pc.get_normal(normal1, normal2, scaling, rot, dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
    specular = specular # (N, 3)
    roughness = roughness # (N, 1)
    # if pbr:
    #     color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse_color[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

    # 相机空间下的法线
    # normal = normal @ viewpoint_camera.world_view_transform[:3, :3]

    # 修改后（压缩维度 + 恢复梯度）
    # if pbr:
    #     colors_precomp = color.squeeze()  # [1,1,N,3] -> [N,3]
    # else:
    # colors_precomp = diffuse_color

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # render_extras = {}
    # normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
    # render_extras.update({"normal": normal_normed})
    # if delta_normal_norm is not None:
    #     render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})
   
    
    # out_extras = {}
    # for k in render_extras.keys():
    #     if render_extras[k] is None: continue
    #     image = rasterizer(
    #         means3D = xyz,
    #         means2D = screenspace_points,
    #         shs = None,
    #         colors_precomp = render_extras[k],
    #         opacities = opacity,
    #         scales = scaling,
    #         rotations = rot,
    #         cov3D_precomp = None)[0]
    #     out_extras[k] = image   

    #out_extras["normal"] = (out_extras["normal"] - 0.5) * 2. # range (0, 1) -> (-1, 1)
    
    
    r3dg_raster_settings = RasterSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(viewpoint_camera.intrinsics[0, 2]),
        cy=float(viewpoint_camera.intrinsics[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=r3dg_raster_settings)

    normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
    render_extras = {"normal": normal_normed}
    render_extras.update({
        "pos": xyz,
        "diffuse": diffuse_color, 
        "specular": specular, 
        "roughness": roughness, 
        })
    
    features = torch.cat([f for f in render_extras.values()], dim=-1)
    out_extras = {}
    (_, _, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, distortion, radii) = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = torch.ones_like(screenspace_points),
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None, 
        features=features)
    out_values = torch.split(rendered_feature, [i.shape[1] for i in render_extras.values()])
    out_extras = {
        k: v for k, v in zip(render_extras.keys(), out_values)
    }
    out_extras["depth"] = rendered_depth
    out_extras["rendered_surface_xyz"] = rendered_surface_xyz
    out_extras["distortion"] = distortion
    out_extras["normal"] = (out_extras["normal"] - 0.5) * 2
    out_extras["normal_ref"] = rendered_pseudo_normal
    out_extras["alpha"] = rendered_opacity

    defer_normal = F.normalize(out_extras['normal'].permute(1, 2, 0), dim=-1).reshape(1, 1, -1, 3)
    rendered_image, brdf_pkg = pc.brdf_mlp.shade(out_extras['pos'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            defer_normal, 
                                            out_extras['diffuse'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            out_extras['specular'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            out_extras['roughness'].permute(1, 2, 0)[:, :, 0].reshape(1, 1, -1, 1), 
                                        viewpoint_camera.camera_center[None, None, :].repeat(
                                            int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 1
                                            ).reshape(1, 1, -1, 3))

    if is_training:
        out = {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling,
                    }
  
    else:
        out = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    out.update(out_extras)

    
    return out


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0