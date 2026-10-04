import torch
import torch.nn as nn
from nets.module import PERSONA, GaussianRenderer
from utils.smflix import smflix
import copy

class Model(nn.Module):
    # viewer only: the avatar (persona), its Gaussian renderer and the SMFLIX layer for the mesh overlay
    def __init__(self, persona):
        super(Model, self).__init__()
        self.persona = persona
        self.gaussian_renderer = GaussianRenderer()
        self.smflix_layer = copy.deepcopy(smflix.layer)

    def get_smflix_outputs(self, smflix_param, cam_param):
        root_pose = smflix_param['root_pose'].view(1,3)
        body_pose = smflix_param['body_pose'].view(1,(len(smflix.joint['part_idx']['body'])-1)*3)
        jaw_pose = smflix_param['jaw_pose'].view(1,3)
        leye_pose = smflix_param['leye_pose'].view(1,3)
        reye_pose = smflix_param['reye_pose'].view(1,3)
        lhand_pose = smflix_param['lhand_pose'].view(1,len(smflix.joint['part_idx']['lhand'])*3)
        rhand_pose = smflix_param['rhand_pose'].view(1,len(smflix.joint['part_idx']['rhand'])*3)
        expr = smflix_param['expr'].view(1,smflix.expr_param_dim)
        eyelid = smflix_param['eyelid'].view(1,2)
        trans = smflix_param['trans'].view(1,3)

        shape = self.persona.shape_param[None]
        betas_head = self.persona.betas_head[None]
        face_offset = self.persona.face_offset[None].float().cuda()
        joint_offset = smflix.get_joint_offset(self.persona.joint_offset[None])

        # camera coordinate system
        output = self.smflix_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, eyelid=eyelid, betas=shape, betas_head=betas_head, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        vert, kpt = output.vertices[0], output.joints[0,smflix.kpt['idx'],:]

        # camera coordinate system -> world coordinate system
        vert = torch.matmul(torch.inverse(cam_param['R']), (vert - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        kpt = torch.matmul(torch.inverse(cam_param['R']), (kpt - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        return vert, kpt

def get_model():
    persona = PERSONA()
    with torch.no_grad():
        persona.init()
    return Model(persona)
