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

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

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
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color (now as diffuse color)
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            diffuse_color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            diffuse_color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            diffuse_color = pc.get_color_mlp(cat_local_view)
        else:
            diffuse_color = pc.get_color_mlp(cat_local_view_wodist)
    diffuse_color = diffuse_color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's normal
    # if pc.add_color_dist:
    #     normal = pc.get_normal_mlp(cat_local_view)
    # else:
    #     normal = pc.get_normal_mlp(cat_local_view_wodist)
    # normal = normal.reshape([anchor.shape[0]*pc.n_offsets, 3])
    # normal = torch.nn.functional.normalize(normal, dim=-1)  # 确保法线是单位向量

    specular = pc.get_specular_mlp(cat_local_view_wodist)
    specular = specular.reshape([anchor.shape[0]*pc.n_offsets, 3])

    roughness = pc.get_roughness_mlp(cat_local_view_wodist)
    roughness = roughness.reshape([anchor.shape[0]*pc.n_offsets, 1])

    features_rest = pc.get_features_rest_mlp(cat_local_view_wodist)
    features_rest = features_rest.reshape([anchor.shape[0]*pc.n_offsets, 3])

    normal1 = pc.get_normal1_mlp(cat_local_view_wodist)
    normal1 = normal1.reshape([anchor.shape[0]*pc.n_offsets, 3])
    normal2 = pc.get_normal2_mlp(cat_local_view_wodist)
    normal2 = normal2.reshape([anchor.shape[0]*pc.n_offsets, 3])

    
    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, diffuse_color, scale_rot, offsets, normal1 ,normal2, specular,roughness,features_rest], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, diffuse_color, scale_rot, offsets, normal1 ,normal2, specular,roughness,features_rest = masked.split([6, 3, 3, 7, 3,  3, 3, 3, 1, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets


    # 处理法线

    # 返回用于光照计算的组件
    if is_training:
        return xyz, diffuse_color, opacity, scaling, rot, neural_opacity, mask , normal1, normal2, specular, roughness, features_rest
    else:
        return xyz, diffuse_color, opacity, scaling, rot , normal1, normal2, specular, roughness, features_rest


def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # 是否训练了颜色mlp
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, diffuse_color, opacity, scaling, rot, neural_opacity, mask, normal1, normal2, specular, roughness, features_rest = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, diffuse_color, opacity, scaling, rot, normal1, normal2, specular, roughness, features_rest = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # # 计算光照
    # # 使用可学习的光源方向
    # light_dir = pc.get_light_direction()
    # light_dir = light_dir.unsqueeze(0).expand(xyz.shape[0], -1)  # 扩展到所有点

    # # 计算漫反射
    # diffuse = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
    # diffuse = diffuse * material[:, 0:1]  # 应用漫反射系数
 
    # # 计算高光 (Blinn-Phong)
    # view_dir = -ob_view.unsqueeze(1).repeat(1, pc.n_offsets, 1)  # [N, n_offsets, 3]
    # view_dir = view_dir.view(-1, 3)[mask] if is_training else view_dir.view(-1, 3)  # 只保留mask为True的view_dir
    # view_dir = torch.nn.functional.normalize(view_dir, dim=-1)  # 确保是单位向量
    
    # half_dir = torch.nn.functional.normalize(light_dir + view_dir, dim=-1)
    # specular = torch.pow(torch.clamp(torch.sum(normal * half_dir, dim=-1, keepdim=True), 0.0, 1.0), 
    #                     material[:, 2:3] * 100.0)  # 高光指数
    # specular = specular * material[:, 1:2]  # 应用高光系数

    # # 计算最终颜色
    # color = diffuse_color * (diffuse + 0.1) + specular  # 0.1是环境光

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    


    # 执行这里的shading
    gb_pos = xyz # (N*k, 3) 

    # 计算每个高斯的视角方向
    dir_pp = (gb_pos - viewpoint_camera.camera_center.repeat(gb_pos.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N*k, 3)

    
    normal, delta_normal = pc.get_normal(normal1, normal2,scaling,rot,dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
    delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)   

    specular  = specular # (N*k, 3)
    roughness = roughness # (N*k, 1)
    features_rest = features_rest # (N*k, 3)

    #color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])
    color = diffuse_color

    # colors_precomp = color.squeeze() # (N, 3)
    # diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
    # specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 

    # if pc.brdf_dim>0:
    #     # 残差色
    #     shs_view = pc.get_brdf_features.view(-1, 3, (pc.brdf_dim+1)**2)
        
    #     # 视角方向
    #     dir_pp = (xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)) 
    #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #     sh2rgb = eval_sh(pc.brdf_dim, shs_view, dir_pp_normalized)
    #     color_delta = sh2rgb
    #     colors_precomp += color_delta



    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
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
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # 渲染法线图
    # Calculate Gaussians projected depth
    p_hom = torch.cat([xyz, torch.ones_like(xyz[...,:1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
    p_view = p_view[...,:3,:]
    depth = p_view.squeeze()[...,2:3]
    depth = depth.repeat(1,3)


    render_extras = {"depth": depth}
   
    normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
    render_extras.update({"normal": normal_normed})
    if delta_normal_norm is not None:
        render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})
   
    
    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)[0]
        out_extras[k] = image   

    out_extras["normal"] = (out_extras["normal"] - 0.5) * 2. # range (0, 1) -> (-1, 1)
    
    # Rasterize visible Gaussians to alpha mask image. 
    raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(xyz) 


    out_extras["alpha"] =  rasterizer_alpha(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)[0]
    
    # Render normal from depth image, and alpha blend with the background. 
    # 渲染法线图
    out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color=bg_color, alpha=out_extras["alpha"][0])
    
    normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.

    train_out = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    train_out.update(out_extras)

    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            }
    out.update(out_extras)

    if is_training:
        return train_out
    else:
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

