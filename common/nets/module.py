import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import get_fov, get_view_matrix, get_proj_matrix
from utils.smflix import smflix
from utils.smflix_lib.smflix.lbs import batch_rigid_transform
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from config import cfg
import copy
import os.path as osp

class PERSONA(nn.Module):
    def __init__(self):
        super(PERSONA, self).__init__()
        self.smflix_layer = copy.deepcopy(smflix.layer).cuda()
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        self.mean_offset_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(smflix.joint['part_idx']['body'])-1)*6, 128, 128, 128, 3], relu_final=False, use_gn=True)

    def init(self):
        ## static assets
        # upsample mesh and other assets
        xyz, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smflix_layer.lbs_weights.float()
        expr_dirs = self.smflix_layer.expr_dirs.view(smflix.vertex_num,3*smflix.expr_param_dim)
        eyeliddirs = self.smflix_layer.eyeliddirs.float().view(smflix.vertex_num,3*2) # SMFLIX eyelid blend shapes [left, right]
        is_rhand, is_lhand, is_face_expr = torch.zeros((smflix.vertex_num,1)).float().cuda(), torch.zeros((smflix.vertex_num,1)).float().cuda(), torch.zeros((smflix.vertex_num,1)).float().cuda()
        is_rhand[smflix.rhand_vertex_idx], is_lhand[smflix.lhand_vertex_idx], is_face_expr[smflix.expr_vertex_idx] = 1.0, 1.0, 1.0
        is_expr_boundary, is_mouth_in, is_mouth_out = smflix.is_expr_boundary[:,None].float().cuda(), smflix.is_mouth_in[:,None].float().cuda(), smflix.is_mouth_out[:,None].float().cuda()
        _, skinning_weight, expr_dirs, eyeliddirs, is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out = smflix.upsample_mesh(torch.ones((smflix.vertex_num,3)).float().cuda(), [skinning_weight, expr_dirs, eyeliddirs, is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out]) # upsample with dummy vertex

        expr_dirs = expr_dirs.view(smflix.vertex_num_upsampled,3,smflix.expr_param_dim)
        eyeliddirs = eyeliddirs.view(smflix.vertex_num_upsampled,3,2)
        is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out = is_rhand[:,0] == 1, is_lhand[:,0] == 1, is_face_expr[:,0] == 1, is_expr_boundary[:,0] == 1, is_mouth_in[:,0] == 1, is_mouth_out[:,0] == 1
        is_eye = 0
        for name in ('R_Eye', 'L_Eye'):
            is_eye += torch.max(skinning_weight, 1)[1]==smflix.joint['name'].index(name)
        is_eye = is_eye > 0
        is_face = 0
        for name in ('Head', 'L_Eye', 'R_Eye', 'Jaw'):
            is_face += torch.max(skinning_weight, 1)[1]==smflix.joint['name'].index(name)
        is_face = is_face > 0
        # [train_sr] head Gaussians (scalp / hair, ears and neck included): the rows the head pass updates
        is_head = 0
        for name in ('Head', 'L_Eye', 'R_Eye', 'Jaw', 'Neck'):
            is_head += torch.max(skinning_weight, 1)[1]==smflix.joint['name'].index(name)
        is_head = is_head > 0

        # [train_sr] rows each pass of train_sr.py updates: the head pass owns is_head (head, eyes, jaw, neck),
        # the body pass everything else. (V,1) floats, applied to the gradient and to the update in train_sr.py
        row_mask_head = is_head[:,None].float()
        self.row_mask_head, self.row_mask_body = row_mask_head, 1 - row_mask_head

        self.pos_enc_vert = xyz
        self.skinning_weight_orig = skinning_weight
        self.expr_dirs = expr_dirs
        self.eyeliddirs = eyeliddirs
        self.is_rhand = is_rhand
        self.is_lhand = is_lhand
        self.is_face_expr = is_face_expr
        self.is_expr_boundary = is_expr_boundary
        self.is_mouth_in = is_mouth_in
        self.is_mouth_out = is_mouth_out
        self.is_hand = (is_rhand+is_lhand) > 0
        self.is_face = is_face
        self.is_eye = is_eye
        self.is_head = is_head  # [train_sr] head pass rows
        self.register_buffer('shape_param', smflix.shape_param)
        self.register_buffer('face_offset', smflix.face_offset)
        self.register_buffer('joint_offset', smflix.joint_offset)
        self.register_buffer('betas_head', smflix.betas_head) # SMFLIX head shape. saved with the checkpoint like shape_param

        # load diffused skinning weight
        sw = np.load(osp.join(cfg.sw_path, 'diffused_skinning_weights.npz')) # the field and its grid coords, compressed
        field = sw['field'] # (D_x, D_y, D_z, smflix.joint['num'])
        coord_x, coord_y, coord_z = sw['x'], sw['y'], sw['z'] # each: (D_x,), (D_y,), (D_z,)
        self.skinning_weight = torch.FloatTensor(field).cuda()
        self.skinning_weight_grid = torch.stack([torch.FloatTensor(coord_x), torch.FloatTensor(coord_y), torch.FloatTensor(coord_z)],1).cuda()

        ## optimizable assets
        # initialize scale
        xyz, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=True)
        points = knn_points(xyz[None,:,:], xyz[None,:,:], K=4, return_nn=True)
        dist = torch.sum((xyz[:,None,:] - points.knn[0,:,1:,:])**2,2).mean(1) # average of distances to top-3 closest points (exactly same as https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L134)
        dist = torch.clamp_min(dist, 0.0000001)
        self.mean_offset = nn.Parameter(torch.zeros((smflix.vertex_num_upsampled,3)).float().cuda())
        self.scale = nn.Parameter(torch.log(torch.sqrt(dist))[:,None])

        # initialize from unwrapped texture
        uv = torch.zeros((smflix.vertex_num,2)).float().cuda()
        uv[smflix.face,:] = torch.FloatTensor(smflix.vertex_uv[smflix.face_uv,:]).cuda()
        rgb = F.grid_sample(smflix.texture[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb_aux = F.grid_sample(smflix.texture_gen[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        seg = F.grid_sample(smflix.seg[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        pairs = torch.stack([torch.LongTensor(smflix.face).view(-1), torch.LongTensor(smflix.face_uv).view(-1)], dim=1).cuda()
        unique_pairs = pairs.unique(dim=0)
        is_seam = (torch.bincount(unique_pairs[:, 0], minlength=smflix.vertex_num)[:,None] > 1).float()
        _, uv, is_seam, rgb_smoothed, rgb_aux_smoothed, seg_smoothed = smflix.upsample_mesh(torch.ones((smflix.vertex_num,3)).float().cuda(), [uv, is_seam, rgb, rgb_aux, seg]) # upsample with dummy vertex
        is_seam = (is_seam > 0).float()

        # rgb
        rgb = F.grid_sample(smflix.texture[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb = rgb*(1-is_seam) + rgb_smoothed*is_seam
        rgb = torch.logit(rgb, eps=1e-4)
        self.rgb = nn.Parameter(rgb.clone())
        self.rgb_orig = rgb.clone()

        rgb_aux = F.grid_sample(smflix.texture_gen[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb_aux = rgb_aux*(1-is_seam) + rgb_aux_smoothed*is_seam
        rgb_aux = torch.logit(rgb_aux, eps=1e-4)
        self.rgb_aux = nn.Parameter(rgb_aux.clone())

        # part segmentation
        seg = F.grid_sample(smflix.seg[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        seg = seg*(1-is_seam) + seg_smoothed*is_seam
        seg = torch.logit(seg, eps=1e-4)
        self.seg = nn.Parameter(seg)

    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.mean_offset], 'name': 'mean_offset', 'lr': cfg.lr},
            {'params': [self.scale], 'name': 'scale', 'lr': cfg.lr*10},
            {'params': [self.rgb], 'name': 'rgb', 'lr': cfg.lr*10},
            {'params': [self.rgb_aux], 'name': 'rgb_aux', 'lr': cfg.lr*10},
            {'params': [self.seg], 'name': 'seg', 'lr': cfg.lr*10},
            {'params': [self.triplane], 'name': 'triplane', 'lr': cfg.lr},
            {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net', 'lr': cfg.lr}
        ]
        return optimizable_params



    # [train_sr] head pass: per-row avatar params only, at cfg.lr (= cfg.sr_lr), without rgb_aux.
    # Model.__init__ freezes triplane and mean_offset_offset_net for BOTH SR passes; their forward outputs are retained.
    def get_optimizable_params_sr_head(self):
        return [
            {'params': [self.mean_offset], 'name': 'mean_offset', 'lr': cfg.lr},
            {'params': [self.scale], 'name': 'scale', 'lr': cfg.lr*10},
            {'params': [self.rgb], 'name': 'rgb', 'lr': cfg.lr*10},
            {'params': [self.seg], 'name': 'seg', 'lr': cfg.lr*10},
        ]

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info, return_joint=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smflix.neutral_body_pose.view(1,-1).cuda() # 大 pose
        zero_hand_pose = torch.zeros((1,len(smflix.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smflix.expr_param_dim)).float().cuda()
        zero_eyelid = torch.zeros((1,2)).float().cuda() # eyelid is applied per frame in forward(), not in the template
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smflix.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            betas_head = self.betas_head[None,:]
            face_offset = self.face_offset[None,:,:]
            joint_offset = self.joint_offset[None,:,:]
        else:
            shape_param = torch.zeros((1,smflix.shape_param_dim)).float().cuda()
            betas_head = torch.zeros((1,smflix.betas_head_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smflix_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, eyelid=zero_eyelid, betas=shape_param, betas_head=betas_head, face_offset=face_offset, joint_offset=joint_offset)

        vert_neutral_pose = output.vertices[0] # 大 pose human
        vert_neutral_pose_upsampled = smflix.upsample_mesh(vert_neutral_pose) # 大 pose human
        joint_neutral_pose = output.joints[0][:smflix.joint['num'],:] # 大 pose human
        if not return_joint:
            return vert_neutral_pose_upsampled, vert_neutral_pose
        else:
            return vert_neutral_pose_upsampled, vert_neutral_pose, joint_neutral_pose

    def get_zero_pose_human(self, return_vert=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smflix.joint['part_idx']['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smflix.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smflix.expr_param_dim)).float().cuda()
        zero_eyelid = torch.zeros((1,2)).float().cuda()
        shape_param = self.shape_param[None,:]
        betas_head = self.betas_head[None,:]
        face_offset = self.face_offset[None,:,:].float().cuda()
        joint_offset = smflix.get_joint_offset(self.joint_offset[None,:,:])
        output = self.smflix_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, eyelid=zero_eyelid, betas=shape_param, betas_head=betas_head, face_offset=face_offset, joint_offset=joint_offset)

        joint_zero_pose = output.joints[0][:smflix.joint['num'],:] # zero pose human
        if not return_vert:
            return joint_zero_pose
        else:
            vert_zero_pose = output.vertices[0] # zero pose human
            vert_zero_pose_upsampled = smflix.upsample_mesh(vert_zero_pose) # zero pose human
            return vert_zero_pose_upsampled, vert_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, joint_zero_pose, smflix_param, jaw_zero_pose):
        # 1. 大 pose -> zero pose
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smflix.neutral_body_pose.view(len(smflix.joint['part_idx']['body'])-1,3).cuda() # 大 pose
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smflix.neutral_jaw_pose.view(1,3).cuda() # open mouth
        zero_hand_pose = torch.zeros((len(smflix.joint['part_idx']['lhand']),3)).float().cuda()
        pose = torch.cat((zero_pose, neutral_body_pose, jaw_pose, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) # follow smflix.joint['name']
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_1 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smflix_layer.parents)
        transform_mat_joint_1 = torch.inverse(transform_mat_joint_1[0])

        # 2. zero pose -> image pose
        root_pose = smflix_param['root_pose'].view(1,3)
        body_pose = smflix_param['body_pose'].view(len(smflix.joint['part_idx']['body'])-1,3)
        jaw_pose = smflix_param['jaw_pose'].view(1,3)
        leye_pose = smflix_param['leye_pose'].view(1,3)
        reye_pose = smflix_param['reye_pose'].view(1,3)
        lhand_pose = smflix_param['lhand_pose'].view(len(smflix.joint['part_idx']['lhand']),3)
        rhand_pose = smflix_param['rhand_pose'].view(len(smflix.joint['part_idx']['rhand']),3)
        trans = smflix_param['trans'].view(1,3)
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # follow smflix.joint['name']
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smflix_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        translation_mat = torch.zeros((smflix.joint['num'],4,4)).float().cuda()
        translation_mat[:,:3,3] = trans # global translation
        transform_mat_joint_2 += translation_mat

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint

    def get_skinning_weight(self, mean_3d, scale):
        # offsets to consider isotropic Gaussian scales (+-2 sigma)
        # weights based on Gaussian PDF values:
        # +-1 sigma ≈ 0.6065 (~0.5 for simplicity), +-2 sigma ≈ 0.1353 (~0.3 for simplicity)
        offsets = torch.FloatTensor([
            [0,0,0],
            [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],
            [0,0,1],[0,0,-1],
            [2,0,0],[-2,0,0],
            [0,2,0],[0,-2,0],
            [0,0,2],[0,0,-2],
        ]).cuda() # (offset_num, 3)
        weights = torch.FloatTensor([1.0] + [0.5 for _ in range(6)] + [0.3 for _ in range(6)]).cuda() # (offset_num,)
        offset_num = offsets.shape[0]

        # prepare sampling coordinates by normalizing them to [-1,1]
        def normalize_coords(points, x, y, z):
            nx = 2 * (points[:, 0] - x.min()) / (x.max() - x.min()) - 1
            ny = 2 * (points[:, 1] - y.min()) / (y.max() - y.min()) - 1
            nz = 2 * (points[:, 2] - z.min()) / (z.max() - z.min()) - 1
            return torch.stack([nx, ny, nz], dim=1)
        xyz = mean_3d[:,None,:] + offsets[None,:,:] * scale[:,None,:] # (smflix.vertex_num_upsampled, offset_num, 3)
        xyz = xyz.view(-1,3)
        xyz = normalize_coords(xyz, self.skinning_weight_grid[:,0], self.skinning_weight_grid[:,1], self.skinning_weight_grid[:,2])

        # trilinear interpolation from diffused skinning weight field
        grid = xyz.view(1, smflix.vertex_num_upsampled*offset_num, 1, 1, 3)
        field = self.skinning_weight.permute(3,2,1,0)[None] # (1, smflix.joint['num'], D_z, D_y, D_x)
        skinning_weight = F.grid_sample(field, grid, mode='bilinear', align_corners=True) # (1, smflix.joint['num'], smflix.vertex_num_upsampled*offset_num, 1, 1)
        skinning_weight = skinning_weight[0,:,:,0,0].permute(1,0).reshape(smflix.vertex_num_upsampled, offset_num, smflix.joint['num'])
        skinning_weight = torch.clamp(skinning_weight, min=0)

        # weighted sum the sampled skinning weight
        skinning_weight = (skinning_weight * weights[None,:,None]).sum(dim=1) / (weights.sum() + 1e-8) # (smflix.vertex_num_upsampled, smflix.joint['num'])
        skinning_weight = F.normalize(torch.clamp(skinning_weight, min=0), p=1, dim=1)

        # for hands and face, assign original vertex index to use skinning weight of the original vertex
        mask = ((self.is_hand + self.is_face) > 0).float()[:,None]
        skinning_weight = self.skinning_weight_orig*mask + skinning_weight*(1-mask)
        return skinning_weight

    def lbs(self, xyz, transform_mat_vertex):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smflix.vertex_num_upsampled,4)[:,:3]
        return xyz

    def extract_tri_feature(self):
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_vert
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)

        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smflix.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smflix.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smflix.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smflix.vertex_num_upsampled, cfg.triplane_shape[0]*3
        return tri_feat

    def get_mean_offset_offset(self, tri_feat, smflix_param, joint_idxs):
        # pose from smflix parameters (only use body pose as face/hand poses are not diverse in the training set)
        body_pose = smflix_param['body_pose'].view(len(smflix.joint['part_idx']['body'])-1,3)

        # combine pose features with triplane feature
        # for pose, take 4-ring joints for each vertex for better generalizability
        adj_joint_mask = smflix.adj_joint_mask.cuda()[joint_idxs,:] # smflix.vertex_num_upsampled, smflix.joint['num']
        adj_joint_mask = adj_joint_mask[:,smflix.joint['part_idx']['body']][:,1:] # take only body joint mask and exclude the root joint.
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).view(1,len(smflix.joint['part_idx']['body'])-1,6).repeat(smflix.vertex_num_upsampled,1,1) # without root pose
        pose = (pose * adj_joint_mask[:,:,None]).view(smflix.vertex_num_upsampled, (len(smflix.joint['part_idx']['body'])-1)*6) # for joints that are not included in the 4-ring, make poses zero

        # forward to geometry networks
        feat = torch.cat((tri_feat, pose.detach()),1)
        mean_offset_offset = self.mean_offset_offset_net(feat)/100 # pose-dependent mean offset of Gaussians

        # for hands, eyes, and face, do not use offsets as they have very small pose-dependent deformations
        mask = ((self.is_hand + self.is_face_expr + self.is_eye) > 0)[:,None].float()
        if cfg.head_no_pose_offset:  # also for the whole head (is_head: Head/Eye/Jaw/Neck Gaussians) # Added by Oh
            mask = ((mask[:,0] > 0) + self.is_head)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        return mean_offset_offset

    def forward(self, smflix_param, cam_param):
        vert_neutral_pose, vert_neutral_pose_wo_upsample = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()

        # get geometry Gaussian features
        mean_offset = self.mean_offset
        scale = torch.exp(self.scale).repeat(1,3)
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smflix.vertex_num_upsampled,1,1)) # constant
        opacity = torch.ones((smflix.vertex_num_upsampled,1)).float().cuda() # constant
        rgb = torch.sigmoid(self.rgb)
        mean_3d = vert_neutral_pose + mean_offset # 大 pose

        # get skinning weight
        skinning_weight = self.get_skinning_weight(mean_3d, scale.detach())

        # pose-dependent mean offsets
        tri_feat = self.extract_tri_feature()
        joint_idxs = torch.argmax(skinning_weight,1)
        mean_offset_offset = self.get_mean_offset_offset(tri_feat, smflix_param, joint_idxs)
        mean_3d_refined = mean_3d + mean_offset_offset # 大 pose

        # smflix facial expression and eyelid offsets. both are canonical-space blend shapes that the SMFLIX
        # layer would add right before skinning (smflix_lib/smflix/lbs.py), so add them here the same way
        smflix_expr_offset = (smflix_param['expr'][None,None,:] * self.expr_dirs).sum(2)
        smflix_eyelid_offset = (smflix_param['eyelid'].view(2)[None,None,:] * self.eyeliddirs).sum(2)
        smflix_face_offset = smflix_expr_offset + smflix_eyelid_offset
        vert = vert_neutral_pose + smflix_face_offset # 大 pose
        mean_3d = mean_3d + smflix_face_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smflix_face_offset # 大 pose

        # get skinning weight
        skinning_weight_refined = self.get_skinning_weight(mean_3d_refined, scale.detach())

        # forward kinematics and lbs
        transform_mat_joint = self.get_transform_mat_joint(joint_zero_pose, smflix_param, jaw_zero_pose=True) # follow jaw_pose of the vert_neutral_pose
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smflix.joint['num'],16)).view(smflix.vertex_num_upsampled,4,4)
        vert = self.lbs(vert, transform_mat_vertex) # posed with smflix_param
        mean_3d = self.lbs(mean_3d, transform_mat_vertex) # posed with smflix_param
        transform_mat_vertex = torch.matmul(skinning_weight_refined, transform_mat_joint.view(smflix.joint['num'],16)).view(smflix.vertex_num_upsampled,4,4)
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex) # posed with smflix_param

        # camera coordinate system -> world coordinate system
        vert = torch.matmul(torch.inverse(cam_param['R']), (vert - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)

        # Gaussians and offsets
        assets = {'mean_3d': mean_3d, 'opacity': opacity, 'scale': scale, 'rotation': rotation, 'rgb': rgb}
        assets_refined = {'mean_3d': mean_3d_refined, 'opacity': opacity, 'scale': scale, 'rotation': rotation, 'rgb': rgb}
        offsets = {'mean_offset': mean_offset, 'mean_offset_offset': mean_offset_offset}
        return assets, assets_refined, offsets, vert_neutral_pose, vert

class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()

    def forward(self, gaussian_assets, img_shape, cam_param, bg=torch.ones((3)).float().cuda()):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            kernel_size=0.1,
            subpixel_offset=torch.zeros((img_shape[0],img_shape[1],2)).float().cuda(),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()

        # rasterize visible Gaussians to image and obtain their radius (on screen)
        render_img, radius = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None
        )

        out = {'img': render_img, 'mean_2d': mean_2d, 'is_vis': radius > 0, 'radius': radius}
        return out

class SMFLIXParamDict(nn.Module):
    def __init__(self):
        super(SMFLIXParamDict, self).__init__()

    # initialize SMFLIX parameters of all frames
    def init(self, smflix_params):
        _smflix_params = {}
        for split in smflix_params.keys():
            _smflix_params[str(split)] = nn.ParameterDict({})
            for frame_idx in smflix_params[split].keys():
                _smflix_params[str(split)][str(frame_idx)] = nn.ParameterDict({})
                # 'eyelid' (2-dim, [left, right]) is optimized per frame like 'expr'
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'eyelid', 'trans']:
                    if 'pose' in param_name:
                        _smflix_params[str(split)][str(frame_idx)][param_name] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(smflix_params[split][frame_idx][param_name].cuda())))
                    else:
                        _smflix_params[str(split)][str(frame_idx)][param_name] = nn.Parameter(smflix_params[split][frame_idx][param_name].cuda())

        self.smflix_params = nn.ParameterDict(_smflix_params)

    def get_optimizable_params(self):
        optimizable_params = []
        for split in self.smflix_params.keys():
            for frame_idx in self.smflix_params[split].keys():
                for param_name in self.smflix_params[split][frame_idx].keys():
                    if 'hand_pose' in param_name:
                        lr = cfg.smflix_param_lr / 10
                    else:
                        lr = cfg.smflix_param_lr

                    optimizable_params.append({'params': [self.smflix_params[split][frame_idx][param_name]], 'name': 'smflix_param_' + param_name + '_' + split + '_' + frame_idx, 'lr': lr})
        return optimizable_params

    # [head_anchor_closed] the closed-eye anchor is captured frame 0 with another eyelid. dataset.py builds it from the preprocess
    # fit. Copy the active captured annotation so both anchors share one pose; keep the closed-eye eyelid value.
    def sync_closed_anchor(self):
        if 'captured_closed' not in self.smflix_params or 'captured' not in self.smflix_params:
            return
        src, dst = self.smflix_params['captured']['0'], self.smflix_params['captured_closed']['0']
        with torch.no_grad():
            for name in dst.keys():
                if name != 'eyelid':
                    dst[name].copy_(src[name])

    def forward(self, splits, frame_idxs):
        out = []
        for split, frame_idx in zip(splits,frame_idxs):
            split = str(split)
            frame_idx = str(int(frame_idx))
            smflix_param = {}
            for param_name in self.smflix_params[split][frame_idx].keys():
                if 'pose' in param_name:
                    smflix_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smflix_params[split][frame_idx][param_name]))
                else:
                    smflix_param[param_name] = self.smflix_params[split][frame_idx][param_name]
            out.append(smflix_param)
        return out
