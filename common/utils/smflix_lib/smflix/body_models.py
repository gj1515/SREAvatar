#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn

from .lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, blend_shapes)

from .vertex_ids import vertex_ids as VERTEX_IDS
from .utils import (
    Struct, to_np, to_tensor, Tensor, Array,
    SMPLOutput,
    SMPLHOutput,
    SMFLIXOutput,
    MANOOutput,
    FLAMEOutput,
    find_joint_kin_chain)
from .vertex_joint_selector import VertexJointSelector


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        **kwargs
    ) -> None:
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  ' 10 shape coefficients.')
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age=='kid':
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - data_struct.v_template, axis=2)
            shapedirs = np.concatenate((shapedirs[:, :, :num_betas], v_template_diff), axis=2)
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros(
                    [batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return 'SMPL'

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f'Gender: {self.gender.upper()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
        ]
        return '\n'.join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output


class SMPLLayer(SMPL):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        # Just create a SMPL module without any member variables
        super(SMPLLayer, self).__init__(
            create_body_pose=False,
            create_betas=False,
            create_global_orient=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body.  Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format.  (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                Body pose. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(
                    batch_size, self.NUM_BODY_JOINTS, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3)],
            dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               pose2rot=False)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output


class SMPLH(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
        self, model_path,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_left_hand_pose: bool = True,
        left_hand_pose: Optional[Tensor] = None,
        create_right_hand_pose: bool = True,
        right_hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        gender: str = 'neutral',
        age: str = 'adult',
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' SMPLH model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_left_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the left
                hand. (default = True)
            left_hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            create_right_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            right_hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'SMPLH_{}.{ext}'.format(gender.upper(), ext=ext)
                smplh_path = os.path.join(model_path, model_fn)
            else:
                smplh_path = model_path
            assert osp.exists(smplh_path), 'Path {} does not exist!'.format(
                smplh_path)

            if ext == 'pkl':
                with open(smplh_path, 'rb') as smplh_file:
                    model_data = pickle.load(smplh_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        super(SMPLH, self).__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            data_struct=data_struct,
            batch_size=batch_size, vertex_ids=vertex_ids, gender=gender, age=age,
            use_compressed=use_compressed, dtype=dtype, ext=ext, **kwargs)

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
        right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

        self.np_left_hand_components = left_hand_components
        self.np_right_hand_components = right_hand_components
        if self.use_pca:
            self.register_buffer(
                'left_hand_components',
                torch.tensor(left_hand_components, dtype=dtype))
            self.register_buffer(
                'right_hand_components',
                torch.tensor(right_hand_components, dtype=dtype))

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_struct.hands_meanl)
        else:
            left_hand_mean = data_struct.hands_meanl

        if self.flat_hand_mean:
            right_hand_mean = np.zeros_like(data_struct.hands_meanr)
        else:
            right_hand_mean = data_struct.hands_meanr

        self.register_buffer('left_hand_mean',
                             to_tensor(left_hand_mean, dtype=self.dtype))
        self.register_buffer('right_hand_mean',
                             to_tensor(right_hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_lhand_pose = torch.tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = nn.Parameter(default_lhand_pose,
                                                requires_grad=True)
            self.register_parameter('left_hand_pose',
                                    left_hand_pose_param)

        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_rhand_pose = torch.tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = nn.Parameter(default_rhand_pose,
                                                 requires_grad=True)
            self.register_parameter('right_hand_pose',
                                    right_hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean_tensor = self.create_mean_pose(
            data_struct, flat_hand_mean=flat_hand_mean)
        if not torch.is_tensor(pose_mean_tensor):
            pose_mean_tensor = torch.tensor(pose_mean_tensor, dtype=dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)

        pose_mean = torch.cat([global_orient_mean, body_pose_mean,
                               self.left_hand_mean,
                               self.right_hand_mean], dim=0)
        return pose_mean

    def name(self) -> str:
        return 'SMPL+H'

    def extra_repr(self):
        msg = super(SMPLH, self).extra_repr()
        msg = [msg]
        if self.use_pca:
            msg.append(f'Number of PCA components: {self.num_pca_comps}')
        msg.append(f'Flat hand mean: {self.flat_hand_mean}')
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLHOutput:
        '''
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient, body_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output


class SMPLHLayer(SMPLH):

    def __init__(
        self, *args, **kwargs
    ) -> None:
        ''' SMPL+H as a layer model constructor
        '''
        super(SMPLHLayer, self).__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args,
            **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLHOutput:
        ''' Forward pass for the SMPL+H model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        model_vars = [betas, global_orient, body_pose, transl, left_hand_pose,
                      right_hand_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 21, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
             left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
             right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3)],
            dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output


class SMFLIX(SMPLH):
    '''
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    '''

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12

    def __init__(
        self, model_path: str,
	    kid_template_path: str = '',
        num_expression_coeffs: int = 10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        create_eyelid: bool = False,
        eyelid: Optional[Tensor] = None,
        add_teeth: bool = True,
        use_face_contour: bool = False,
        num_betas_head: int = 300,
        create_betas_head: bool = True,
        betas_head: Optional[Tensor] = None,
        batch_size: int = 1,
        gender: str = 'neutral',
        age: str = 'adult',
        dtype=torch.float32,
        ext: str = 'npz',
        **kwargs
    ) -> None:
        ''' SMPLX model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            create_eyelid: bool, optional
                Deprecated. The eyelid member variable is always created.
                Kept only for call-site compatibility. (default = False)
            eyelid: torch.tensor, optional, Bx2
                The default value for the eyelid member variable [left, right].
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''

        # Load the model
        if osp.isdir(model_path):
            model_fn = 'SMFLIX_NEUTRAL.npz'
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        assert osp.exists(smplx_path), 'Path {} does not exist!'.format(
            smplx_path)

        if ext == 'pkl':
            with open(smplx_path, 'rb') as smplx_file:
                model_data = pickle.load(smplx_file, encoding='latin1')
        elif ext == 'npz':
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            raise ValueError('Unknown extension: {}'.format(ext))

        data_struct = Struct(**model_data)

        super(SMFLIX, self).__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            vertex_ids=VERTEX_IDS['smplx'],
            gender=gender, age=age, ext=ext,
            **kwargs)

        # Fixed by SH Heo (260128) - Apply eyeball Z-offset to v_template
        eyeball_z_offset = -0.0035  # 3.5mm offset (adjustable)
        l_eyeball_ids = list(range(9383, 9929))   # Left eyeball: 9383-9928
        r_eyeball_ids = list(range(9929, 10475))  # Right eyeball: 9929-10474
        eyeball_ids = l_eyeball_ids + r_eyeball_ids
        self.v_template[eyeball_ids, 2] += eyeball_z_offset

        lmk_faces_idx = data_struct.lmk_faces_idx.copy()
        # Fixed by SH Heo (260123) Fixed face landmark positions (eyebrow)
        lmk_faces_idx[0] = 1881  # right_eye_brow1
        lmk_faces_idx[1] = 12420   # right_eye_brow2
        lmk_faces_idx[2] = 9198   # right_eye_brow3
        lmk_faces_idx[3] = 9187  # right_eye_brow4
        lmk_faces_idx[4] = 2970  # right_eye_brow5

        lmk_faces_idx[5] = 11870   # left_eye_brow1
        lmk_faces_idx[6] = 11899  # left_eye_brow2
        lmk_faces_idx[7] = 9285   # left_eye_brow3
        lmk_faces_idx[8] = 10963  # left_eye_brow4
        lmk_faces_idx[9] = 10969  # left_eye_brow5

        lmk_faces_idx[32] = 12987  # outer_lip_right_2
        lmk_faces_idx[33] = 2394   # outer_lip_right_3
        lmk_faces_idx[34] = 19556  # outer_lip_upper_center
        lmk_faces_idx[35] = 11406  # outer_lip_left_3
        lmk_faces_idx[36] = 910    # outer_lip_left_2

        lmk_faces_idx[38] = 11498  # outer_lip_lower_left
        lmk_faces_idx[39] = 985  # outer_lip_lower_center_left
        lmk_faces_idx[40] = 19600  # outer_lip_lower_center
        lmk_faces_idx[41] = 13090   # outer_lip_lower_center_right
        lmk_faces_idx[42] = 13021  # outer_lip_lower_right

        # lip
        # upper
        lmk_faces_idx[44] = 12969  # right_lip_2
        lmk_faces_idx[46] = 887   # left_lip_2

        lmk_faces_idx[45] = 12838  # inner_lip_upper_center

        # lower
        lmk_faces_idx[48] = 11491  # left_lip_3
        lmk_faces_idx[50] = 13018  # right_lip_3

        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords.copy()
        # Fixed by SH Heo (260123) Fixed face landmark barycentric coordinates (eyebrow)
        lmk_bary_coords[0] = [0.13897281, 0.21951239,  0.6415148]  # right_eye_brow1
        lmk_bary_coords[1] = [0.10221249, 0.26428735, 0.63350016]  # right_eye_brow2
        lmk_bary_coords[2] = [0.13297928, 0.08380097, 0.78321974]  # right_eye_brow3
        lmk_bary_coords[3] = [0.35552603, 0.23906546, 0.40540852]  # right_eye_brow4
        lmk_bary_coords[4] = [0.61511322, 0.15279207, 0.2320947]  # right_eye_brow5

        lmk_bary_coords[5] = [0.61511322, 0.2320947, 0.15279207]  # left_eye_brow1
        lmk_bary_coords[6] = [0.35552603, 0.40540852, 0.23906546]  # left_eye_brow2
        lmk_bary_coords[7] = [0.13297928, 0.08380097, 0.78321974]  # left_eye_brow3
        lmk_bary_coords[8] = [0.63350016, 0.26428735, 0.10221249]  # left_eye_brow4
        lmk_bary_coords[9] = [0.13897281, 0.6415148,  0.21951239]  # left_eye_brow5

        lmk_bary_coords[32] = [0.499489, 0.28701395, 0.21349706]  # outer_lip_right_2
        lmk_bary_coords[33] = [0.19279168, 0.4877205,  0.31948782]  # outer_lip_right_3
        lmk_bary_coords[34] = [0.00216507, 0.27413376, 0.72370116]  # outer_lip_upper_center
        lmk_bary_coords[35] = [0.19279168, 0.31948782,  0.4877205]  # outer_lip_left_3
        lmk_bary_coords[36] = [0.499489, 0.21349706, 0.28701395]  # outer_lip_left_2

        lmk_bary_coords[38] = [0.57079009, 0.3319431, 0.09726681]  # outer_lip_lower_left
        lmk_bary_coords[39] = [0.61930516, 0.1397979, 0.24089694]  # outer_lip_lower_center_left
        lmk_bary_coords[40] = [0.00604383, 0.50537664, 0.48857954]  # outer_lip_lower_center
        lmk_bary_coords[41] = [0.61930516, 0.24089694, 0.1397979]  # outer_lip_lower_center_right
        lmk_bary_coords[42] = [0.09726681, 0.3319431, 0.57079009]  # outer_lip_lower_right

        lmk_bary_coords[44] = [0.02819558, 0.0258838, 0.94592062]  # right_lip_2
        lmk_bary_coords[46] = [0.02819558, 0.94592062, 0.0258838]  # left_lip_2
        lmk_bary_coords[45] = [0.00307847, 0.75792832, 0.23899321]  # inner_lip_upper_center
        lmk_bary_coords[48] = [0.23434689, 0.34040873, 0.42524438]  # left_lip_3
        lmk_bary_coords[50] = [0.30888019, 0.45817919, 0.23294063]  # right_lip_3


        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            dynamic_lmk_faces_idx = data_struct.dynamic_lmk_faces_idx
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose,
                                          requires_grad=True)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose,
                                           requires_grad=True)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose,
                                           requires_grad=True)
            self.register_parameter('reye_pose', reye_pose_param)

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM +
                self.EXPRESSION_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  ' 10 shape and 10 expression coefficients.')
            expr_start_idx = 10
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 10)
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(
                num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self._num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.num_expression_coeffs], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression,
                                            requires_grad=True)
            self.register_parameter('expression', expression_param)

        ##-----------------------------------------------------------
        # -- Fixed by SH Heo (251118) - Load eyelid offset data
        ##-----------------------------------------------------------

        package_dir = osp.dirname(osp.abspath(__file__))
        l_eyelid_path = osp.join(package_dir, 'l_eyelid_smplx.npy')
        r_eyelid_path = osp.join(package_dir, 'r_eyelid_smplx.npy')

        l_eyelid_data = np.load(l_eyelid_path)
        r_eyelid_data = np.load(r_eyelid_path)

        l_vertex_ids = torch.from_numpy(
            l_eyelid_data['vertex_id'].astype(np.int64)
        )
        l_vectors = torch.from_numpy(
            np.column_stack([
                l_eyelid_data['x'],
                l_eyelid_data['y'],
                l_eyelid_data['z']
            ]).astype(np.float32)
        )

        r_vertex_ids = torch.from_numpy(
            r_eyelid_data['vertex_id'].astype(np.int64)
        )
        r_vectors = torch.from_numpy(
            np.column_stack([
                r_eyelid_data['x'],
                r_eyelid_data['y'],
                r_eyelid_data['z']
            ]).astype(np.float32)
        )

        # Build dense eyelid blend shapes [V, 3, 2] (channel 0: left, 1: right)
        # The npy vectors point in the eye-opening direction, so negate them
        # so that positive coefficients close the eye (FLAME convention)
        eyeliddirs = torch.zeros([self.v_template.shape[0], 3, 2], dtype=dtype)
        eyeliddirs[l_vertex_ids, :, 0] = -to_tensor(to_np(l_vectors), dtype=dtype)
        eyeliddirs[r_vertex_ids, :, 1] = -to_tensor(to_np(r_vectors), dtype=dtype)
        self.register_buffer('eyeliddirs', eyeliddirs)

        # Create eyelid parameter (learnable)
        if eyelid is None:
            default_eyelid = torch.zeros([batch_size, 2], dtype=dtype)
        else:
            default_eyelid = torch.tensor(eyelid, dtype=dtype)
        eyelid_param = nn.Parameter(default_eyelid, requires_grad=True)
        self.register_parameter('eyelid', eyelid_param)

        # -- FLAME head shape --
        self._num_betas_head = num_betas_head

        shapedirs_head = data_struct.shapedirs_head
        shapedirs_head = shapedirs_head[:, :, :num_betas_head]
        self.register_buffer(
            'shapedirs_head',
            to_tensor(to_np(shapedirs_head), dtype=dtype))

        # flame_ids
        self.register_buffer(
            'flame_ids',
            torch.tensor(data_struct.flame_ids, dtype=torch.long))

        # Fixed by SH Heo (260227) - Add teeth mesh
        if add_teeth:
            self.add_teeth()

        # betas_head parameter
        if create_betas_head:
            if betas_head is None:
                default_betas_head = torch.zeros(
                    [batch_size, num_betas_head], dtype=dtype)
            else:
                if torch.is_tensor(betas_head):
                    default_betas_head = betas_head.clone().detach()
                else:
                    default_betas_head = torch.tensor(betas_head, dtype=dtype)
            self.register_parameter(
                'betas_head',
                nn.Parameter(default_betas_head, requires_grad=True))

    @property
    def num_betas_head(self):
        return self._num_betas_head

    def name(self) -> str:
        return 'SMPL-X'

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    self.left_hand_mean, self.right_hand_mean],
                                   axis=0)

        return pose_mean

    ##-----------------------------------------------------------
    # -- Fixed by SH Heo (260227) - Add teeth mesh to SMFLIX
    ##-----------------------------------------------------------
    def add_teeth(self):
        # SMPLX lip vertex IDs
        vid_lip_outside_ring_upper = [1713, 1715, 1716, 1735, 1696, 1694, 1657, 8987, 2774, 2811, 2813, 2850, 2833, 2832, 2830]
        vid_lip_outside_ring_lower = [1576, 1577, 1773, 1774, 1795, 1802, 1865, 8947, 2948, 2905, 2898, 2881, 2880, 2713, 2712]
        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= mean_dist * 1.5  # how far the teeth are from the lips

        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]]) * 0.25
        v_teeth_upper_edge += torch.tensor([[0, 0, mean_dist]]) * 0.4  # +0.0
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]]) * 0.2
        v_teeth_lower_edge += torch.tensor([[0, 0, mean_dist]]) * 0.2  # -0.4 slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # thickness = mean_dist * 0.5
        thickness = mean_dist * 0.75
        # upper, back
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_upper_edge_back[:, 2] -= thickness  # how thick the teeth are

        # lower, back
        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_lower_edge_back[:, 2] -= thickness  # how thick the teeth are

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
            v_teeth_upper_root_back,  # num_verts_orig + 60-74
            v_teeth_upper_edge_back,  # num_verts_orig + 75-89
            v_teeth_lower_root_back,  # num_verts_orig + 90-104
            v_teeth_lower_edge_back,  # num_verts_orig + 105-119
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120) + num_verts_orig
        vid_teeth_upper = torch.cat(
            [vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat(
            [vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_root_back, vid_teeth_lower_edge_back], dim=0)

        # UV coordinates for teeth (placed in empty region v=0.002~0.018 to avoid overlap with head top in smplx_uv_2023)
        u = torch.linspace(0.1328, 0.2695, 15)
        v = torch.linspace(0.002, 0.018, 7)
        v = v[[3, 2, 0, 1, 3, 4, 6, 5]]
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)
        self.register_buffer('teeth_texcoords', uv)

        # Extend shapedirs (body shape blendshapes — lip mean for teeth)
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (self.shapedirs[vid_lip_outside_ring_upper] + self.shapedirs[vid_lip_outside_ring_lower]) / 2
        self.shapedirs[vid_teeth_upper_root] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_root_back] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root_back] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back] = shape_dirs_mean

        # Extend shapedirs_head (head shape blendshapes — lip mean for teeth)
        self.shapedirs_head = torch.cat([self.shapedirs_head, torch.zeros_like(self.shapedirs_head[:num_verts_teeth])], dim=0)
        shape_dirs_head_mean = (self.shapedirs_head[vid_lip_outside_ring_upper] + self.shapedirs_head[vid_lip_outside_ring_lower]) / 2
        self.shapedirs_head[vid_teeth_upper_root] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_lower_root] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_upper_edge] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_lower_edge] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_upper_root_back] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_upper_edge_back] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_lower_root_back] = shape_dirs_head_mean
        self.shapedirs_head[vid_teeth_lower_edge_back] = shape_dirs_head_mean

        # Extend expr_dirs (expression blendshapes, zero for teeth)
        self.expr_dirs = torch.cat([self.expr_dirs, torch.zeros_like(self.expr_dirs[:num_verts_teeth])], dim=0)

        # Extend eyeliddirs (eyelid blendshapes, zero for teeth)
        self.eyeliddirs = torch.cat([self.eyeliddirs, torch.zeros_like(self.eyeliddirs[:num_verts_teeth])], dim=0)

        # Extend posedirs
        posedirs = self.posedirs.reshape(len(self.parents) - 1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents) - 1) * 9, (num_verts_orig + num_verts_teeth) * 3)  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)

        # lbs_weights: use lip vertex weight averages so teeth track lips
        lbs_upper_mean = self.lbs_weights[vid_lip_outside_ring_upper].mean(dim=0)
        lbs_lower_mean = self.lbs_weights[vid_lip_outside_ring_lower].mean(dim=0)
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)
        self.lbs_weights[vid_teeth_upper_root] = lbs_upper_mean
        self.lbs_weights[vid_teeth_upper_edge] = lbs_upper_mean
        self.lbs_weights[vid_teeth_upper_root_back] = lbs_upper_mean
        self.lbs_weights[vid_teeth_upper_edge_back] = lbs_upper_mean
        self.lbs_weights[vid_teeth_lower_root] = lbs_lower_mean
        self.lbs_weights[vid_teeth_lower_edge] = lbs_lower_mean
        self.lbs_weights[vid_teeth_lower_root_back] = lbs_lower_mean
        self.lbs_weights[vid_teeth_lower_edge_back] = lbs_lower_mean

        # Face topology
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  # 0
            [0, 1, 31],  # 1
            [1, 32, 31],  # 2
            [1, 2, 32],  # 3
            [2, 33, 32],  # 4
            [2, 3, 33],  # 5
            [3, 34, 33],  # 6
            [3, 4, 34],  # 7
            [4, 35, 34],  # 8
            [4, 5, 35],  # 9
            [5, 36, 35],  # 10
            [5, 6, 36],  # 11
            [6, 37, 36],  # 12
            [6, 7, 37],  # 13
            [7, 8, 37],  # 14
            [8, 38, 37],  # 15
            [8, 9, 38],  # 16
            [9, 39, 38],  # 17
            [9, 10, 39],  # 18
            [10, 40, 39],  # 19
            [10, 11, 40],  # 20
            [11, 41, 40],  # 21
            [11, 12, 41],  # 22
            [12, 42, 41],  # 23
            [12, 13, 42],  # 24
            [13, 43, 42],  # 25
            [13, 14, 43],  # 26
            [14, 44, 43],  # 27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ])
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 115],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ])

        self.faces_tensor = torch.cat([self.faces_tensor, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], dim=0)
        self.faces = self.faces_tensor.cpu().numpy()
        self.register_buffer('teeth_faces_uv_idx', torch.cat([f_teeth_upper, f_teeth_lower], dim=0))

    def extra_repr(self):
        msg = super(SMFLIX, self).extra_repr()
        msg = [
            msg,
            f'Number of Expression Coefficients: {self.num_expression_coeffs}'
        ]
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        eyelid: Optional[Tensor] = None,
        betas_head: Optional[Tensor] = None,
        face_offset: Optional[Tensor] = None, # added by Heo
        joint_offset: Optional[Tensor] = None, # added by Heo
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        return_shaped: bool = True,
        **kwargs
    ) -> SMFLIXOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            left_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `left_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            right_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `right_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            eyelid: torch.tensor, optional, shape Bx2
                If given, ignore the member variable `eyelid` and use it
                instead. Eyelid parameters [left, right] for controlling
                eyelid displacement. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose
        expression = expression if expression is not None else self.expression

        # Fixed by SH Heo (251118)
        eyelid = eyelid if eyelid is not None else self.eyelid

        betas_head = betas_head if betas_head is not None else self.betas_head

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                               body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
                               jaw_pose.reshape(-1, 1, 3),
                               leye_pose.reshape(-1, 1, 3),
                               reye_pose.reshape(-1, 1, 3),
                               left_hand_pose.reshape(-1, 15, 3),
                               right_hand_pose.reshape(-1, 15, 3)],
                              dim=1).reshape(-1, 165)

        # Add the mean pose of the model. Does not affect the body, only the
        # hands when flat_hand_mean == False
        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        if betas_head.shape[0] < batch_size: # added by Heo
            betas_head = betas_head.expand(batch_size, -1)
        shape_components = torch.cat([betas, betas_head, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.shapedirs_head, self.expr_dirs], dim=-1)

        # added by Heo (face_offset)
        if face_offset is None:
            vertices = self.v_template
        else:
            vertices = self.v_template + face_offset
        vertices, joints = lbs(shape_components, full_pose, vertices,
                               shapedirs, self.posedirs,
                               self.J_regressor, joint_offset, self.parents, # added by Heo (joint_offset)
                               self.lbs_weights, pose2rot=pose2rot,
                               eyelid=eyelid, eyeliddirs=self.eyeliddirs)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        v_shaped = None
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        output = SMFLIXOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             jaw_pose=jaw_pose,
                             v_shaped=v_shaped,
                             full_pose=full_pose if return_full_pose else None)
        return output


class SMFLIXLayer(SMFLIX):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        # Just create a SMPLX module without any member variables
        super(SMFLIXLayer, self).__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
            *args, **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        eyelid: Optional[Tensor] = None,        # Fixed by SH Heo (251118)
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> SMFLIXOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. It is expected to be in rotation matrix
                format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                Expression coefficients.
                For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            eyelid: torch.tensor, optional, shape Bx2
                Eyelid parameters [left, right] for controlling
                eyelid displacement. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A data class that contains the posed vertices and joints
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl,
                      expression, left_hand_pose, right_hand_pose, jaw_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(
                    batch_size, self.NUM_BODY_JOINTS, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if eyelid is None:
            eyelid = torch.zeros([batch_size, 2], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
             jaw_pose.reshape(-1, 1, 3, 3),
             leye_pose.reshape(-1, 1, 3, 3),
             reye_pose.reshape(-1, 1, 3, 3),
             left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
             right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3)],
            dim=1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               pose2rot=False,
                               eyelid=eyelid, eyeliddirs=self.eyeliddirs,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMFLIXOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             jaw_pose=jaw_pose,
                             transl=transl,
                             full_pose=full_pose if return_full_pose else None)
        return output


class MANO(SMPL):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        create_hand_pose: bool = True,
        hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.num_pca_comps = num_pca_comps
        self.is_rhand = is_rhand
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(
                    model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)

            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        super(MANO, self).__init__(
            model_path=model_path, data_struct=data_struct,
            batch_size=batch_size, vertex_ids=vertex_ids,
            use_compressed=use_compressed, dtype=dtype, ext=ext, **kwargs)

        # add only MANO tips to the extra joints
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS['mano'].values()), dtype=torch.long)

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        hand_components = data_struct.hands_components[:num_pca_comps]

        self.np_hand_components = hand_components

        if self.use_pca:
            self.register_buffer(
                'hand_components',
                torch.tensor(hand_components, dtype=dtype))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean

        self.register_buffer('hand_mean',
                             to_tensor(hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose,
                                           requires_grad=True)
            self.register_parameter('hand_pose',
                                    hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(
            data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        # pose_mean_tensor = torch.tensor(pose_mean, dtype=dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

    def name(self) -> str:
        return 'MANO'

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        return pose_mean

    def extra_repr(self):
        msg = [super(MANO, self).extra_repr()]
        if self.use_pca:
            msg.append(f'Number of PCA components: {self.num_pca_comps}')
        msg.append(f'Flat hand mean: {self.flat_hand_mean}')
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else
                     self.hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum(
                'bi,ij->bj', [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=True,
                               )

        # # Add pre-selected extra joints that might be needed
        # joints = self.vertex_joint_selector(vertices, joints)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(vertices=vertices if return_verts else None,
                            joints=joints if return_verts else None,
                            betas=betas,
                            global_orient=global_orient,
                            hand_pose=hand_pose,
                            full_pose=full_pose if return_full_pose else None)

        return output


class MANOLayer(MANO):
    def __init__(self, *args, **kwargs) -> None:
        ''' MANO as a layer model constructor
        '''
        super(MANOLayer, self).__init__(
            create_global_orient=False,
            create_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args, **kwargs)

    def name(self) -> str:
        return 'MANO'

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if hand_pose is None:
            hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None)

        return output


class FLAME(SMPL):
    NUM_JOINTS = 5
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 0

    def __init__(
        self,
        model_path: str,
        data_struct=None,
        num_expression_coeffs=10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_neck_pose: bool = True,
        neck_pose: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour=False,
        batch_size: int = 1,
        gender: str = 'neutral',
        dtype: torch.dtype = torch.float32,
        ext='pkl',
        **kwargs
    ) -> None:
        ''' FLAME model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_neck_pose: bool, optional
                Flag for creating a member variable for the neck pose.
                (default = False)
            neck_pose: torch.tensor, optional, Bx3
                The default value for the neck pose variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''
        model_fn = f'FLAME_{gender.upper()}.{ext}'
        flame_path = os.path.join(model_path, model_fn)
        assert osp.exists(flame_path), 'Path {} does not exist!'.format(
            flame_path)
        if ext == 'npz':
            file_data = np.load(flame_path, allow_pickle=True)
        elif ext == 'pkl':
            with open(flame_path, 'rb') as smpl_file:
                file_data = pickle.load(smpl_file, encoding='latin1')
        else:
            raise ValueError('Unknown extension: {}'.format(ext))
        data_struct = Struct(**file_data)

        super(FLAME, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            gender=gender,
            ext=ext,
            **kwargs)

        self.use_face_contour = use_face_contour

        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            [], dtype=torch.long)

        if create_neck_pose:
            if neck_pose is None:
                default_neck_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_neck_pose = torch.tensor(neck_pose, dtype=dtype)
            neck_pose_param = nn.Parameter(
                default_neck_pose, requires_grad=True)
            self.register_parameter('neck_pose', neck_pose_param)

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose,
                                          requires_grad=True)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose,
                                           requires_grad=True)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose,
                                           requires_grad=True)
            self.register_parameter('reye_pose', reye_pose_param)

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM +
                self.EXPRESSION_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  ' 10 shape and 10 expression coefficients.')
            expr_start_idx = 10
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 10)
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(
                num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self._num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.num_expression_coeffs], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression,
                                            requires_grad=True)
            self.register_parameter('expression', expression_param)

        # The pickle file that contains the barycentric coordinates for
        # regressing the landmarks
        landmark_bcoord_filename = osp.join(
            model_path, 'flame_static_embedding.pkl')

        with open(landmark_bcoord_filename, 'rb') as fp:
            landmarks_data = pickle.load(fp, encoding='latin1')

        lmk_faces_idx = landmarks_data['lmk_face_idx'].astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = landmarks_data['lmk_b_coords']
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))
        if self.use_face_contour:
            face_contour_path = os.path.join(
                model_path, 'flame_dynamic_embedding.npy')
            contour_embeddings = np.load(face_contour_path,
                                         allow_pickle=True,
                                         encoding='latin1')[()]

            dynamic_lmk_faces_idx = np.array(
                contour_embeddings['lmk_face_idx'], dtype=np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_b_coords = torch.tensor(
                contour_embeddings['lmk_b_coords'], dtype=dtype)
            self.register_buffer(
                'dynamic_lmk_bary_coords', dynamic_lmk_b_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def name(self) -> str:
        return 'FLAME'

    def extra_repr(self):
        msg = [
            super(FLAME, self).extra_repr(),
            f'Number of Expression Coefficients: {self.num_expression_coeffs}',
            f'Use face contour: {self.use_face_contour}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose

        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose

        betas = betas if betas is not None else self.betas
        expression = expression if expression is not None else self.expression

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         jaw_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             neck_pose=neck_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output


class FLAMELayer(FLAME):
    def __init__(self, *args, **kwargs) -> None:
        ''' FLAME as a layer model constructor '''
        super(FLAMELayer, self).__init__(
            create_betas=False,
            create_expression=False,
            create_global_orient=False,
            create_neck_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            *args,
            **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if neck_pose is None:
            neck_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 1, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        joints += transl.unsqueeze(dim=1)
        vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             neck_pose=neck_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output


def build_layer(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPLLayer, SMPLHLayer, SMFLIXLayer, MANOLayer, FLAMELayer]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT
            |-- flame
                |-- FLAME_FEMALE
                |-- FLAME_MALE
                |-- FLAME_NEUTRAL

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPLLayer(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLHLayer(model_path, **kwargs)
    elif model_type.lower() == 'smflix':
        return SMFLIXLayer(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANOLayer(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAMELayer(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')


def create(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPL, SMPLH, SMFLIX, MANO, FLAME]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smflix':
        return SMFLIX(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANO(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAME(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')
