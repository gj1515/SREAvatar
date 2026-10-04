"""Interactive dearpygui viewer for a trained PERSONA (SMFLIX) avatar.

Run from main:
    python viewer.py [--subject_id $SUBJECT_ID] [--test_epoch 4] [--motion_path $MOTION_DIR] [--motion_ckpt $CKPT ...]

Motion sources: per-frame parameters stored in the checkpoint (captured / generated_0_origin / generated_1),
or, when --motion_path is given, pose_track output in the animate.py layout (smplx/params/*.json).
Splits this checkpoint lacks are added from the other snapshots of the same subject (*_original first, then every
<subject>_* experiment), or from the checkpoints given with --motion_ckpt.
The avatar is kept in the fitting camera frame and the viewer moves its own perspective camera around it.
Mouse on the image: left drag = orbit, wheel = zoom, right drag = pan. The Turntable checkbox keeps the camera
circling the avatar around the vertical axis. 'mesh overlay' blends the fitted SMFLIX mesh (shaded gray) over the
render. The window can be resized freely; the render resolution always follows the image area.
"""
import argparse
import json
import math
import os
import os.path as osp
import time
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
import dearpygui.dearpygui as dpg
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_quaternion, rotation_6d_to_matrix

from config import cfg
from base import Tester, rename_body_branch_keys
from nets.layer import rasterize
from utils.smflix import smflix

PANEL_WIDTH = 420          # control panel width (pixels)
MAX_INIT_HEIGHT = 1000     # initial window height cap so that the original resolution still fits the screen
PLAY_FPS = 30              # playback speed cap. never skips frames
TURNTABLE_SPEED = 2.0      # default turntable speed, degrees per rendered frame (one revolution in 180 frames)
NECK_IDX = smflix.joint['name'].index('Neck') - 1   # body_pose has no root joint
FACE_SLIDERS = [('expr', i, 3.0) for i in range(5)] + [('eyelid', i, 1.0) for i in range(2)] + [('jaw_pose', i, 0.5) for i in range(3)] + [('neck', i, 0.5) for i in range(3)] # eyelid [left, right] # Added by Heo


# ---------------------------------------------------------------- helpers ----------------------------------------------------------------
def load_ckpt_params(ckpt_path):
    # per-frame SMFLIX parameters optimized during training, stored as smplx_param_dict.smplx_params.<split>.<frame>.<name>
    # (smflix_param_dict.smflix_params.* in checkpoints from the renamed code, same layout) # Added by Heo
    state = torch.load(ckpt_path, map_location='cpu')['network']
    table = {}
    for key, value in state.items():
        if not key.startswith(('smplx_param_dict.', 'smflix_param_dict.')):
            continue
        _, _, split, frame_idx, name = key.split('.')
        if 'pose' in name:
            value = matrix_to_axis_angle(rotation_6d_to_matrix(value))  # stored as 6D rotation
        table.setdefault(split, {}).setdefault(int(frame_idx), {})[name] = value.reshape(-1).cuda()
    for frames in table.values(): # checkpoints trained before eyelid existed have no eyelid entry. keep the eyes open # Added by Heo
        for param in frames.values():
            param.setdefault('eyelid', torch.zeros(2).float().cuda())
    return {split: [frames[i] for i in sorted(frames)] for split, frames in sorted(table.items())}


def load_avatar_sources(subject_id, test_epoch, motion_ckpt):
    # the avatar's own splits, then those of motion_ckpt or, by default, of its sibling snapshots under avatars/
    sources = load_ckpt_params(osp.join(cfg.root_dir, 'avatars', subject_id, 'snapshot_%d.pth' % test_epoch))
    extra_ckpts = list(motion_ckpt)
    if not extra_ckpts: # default: every avatar of the same subject, *_original first, so all their generated splits are available # Added by Heo
        parts = subject_id.split('_')
        for i in range(len(parts), 0, -1):
            base = osp.join(cfg.root_dir, 'avatars', '_'.join(parts[:i]))
            if osp.isfile(osp.join(base + '_original', 'snapshot_%d.pth' % test_epoch)):
                extra_ckpts = [osp.join(base + '_original', 'snapshot_%d.pth' % test_epoch)]
                extra_ckpts += sorted(x for x in glob(osp.join(base + '_*', 'snapshot_%d.pth' % test_epoch)) if x not in extra_ckpts)
                break
    for ckpt_path in extra_ckpts: # the avatar's own splits win
        print('motion sources also from ' + ckpt_path)
        for split, frames in load_ckpt_params(ckpt_path).items():
            sources.setdefault(split, frames)
    return {k: v for k, v in sources.items() if k == 'captured'} # only the captured motion is offered; the other splits are training-only


def load_motion_params(motion_path):
    # animate.py layout. pose_track motions carry no eyelid parameter. keep the eyes open
    path_list = sorted(glob(osp.join(motion_path, 'smplx', 'params', '*.json')) or glob(osp.join(motion_path, 'smplx_init', '*.json')), key=lambda x: int(osp.basename(x)[:-5])) # or run_smflix.py layout # Added by Heo
    assert path_list, 'No smplx/params/*.json in ' + motion_path
    frames = []
    for path in path_list:
        with open(path) as f:
            param = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}
        param.setdefault('eyelid', torch.zeros(2).float().cuda())
        if frames: # SMFLIX trans is in a per-frame bbox camera, so it jitters under the viewer's fixed camera. hold the first frame's # Added by Heo
            param['trans'] = frames[0]['trans'].clone()
        frames.append(param)
    img_path = osp.join(motion_path, 'images', osp.basename(path_list[0])[:-5] + '.png')
    return {'motion': frames}, img_path


def load_eval_params(eval_path):    # Added by Heo
    with open(eval_path) as f:
        data = json.load(f)
    img_shape = {img['id']: (img['height'], img['width']) for img in data['images']}
    frames = {}
    for ann in data['annotations']:
        param = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in ann['smflix_param'].items()}
        param.setdefault('eyelid', torch.zeros(2).float().cuda())
        cam = ann['cam_param']
        cam_param = {'R': torch.eye(3).float().cuda(), 't': torch.zeros(3).float().cuda(),
                     'focal': torch.FloatTensor(cam['focal']).cuda(), 'princpt': torch.FloatTensor(cam['princpt']).cuda()}
        frames[ann['image_id']] = (param, cam_param, img_shape[ann['image_id']])
    return frames


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


# ---------------------------------------------------------------- avatar ----------------------------------------------------------------
class Avatar:
    """PERSONA.forward split into cached stages: identity-only terms once, pose terms per update, rendering per camera.
    Same math as nets/module.py PERSONA.forward with an identity camera, so world == fitting camera frame."""

    @torch.no_grad()
    def __init__(self, model):
        self.model, self.persona, self.renderer = model, model.persona, model.gaussian_renderer
        self.cam_identity = {'R': torch.eye(3).float().cuda(), 't': torch.zeros(3).float().cuda()}
        self.faces = torch.LongTensor(smflix.face).cuda()
        p = self.persona
        self.vert_neutral_pose, _ = p.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        self.joint_zero_pose = p.get_zero_pose_human()
        self.scale = torch.exp(p.scale).repeat(1, 3)
        self.mean_3d = self.vert_neutral_pose + p.mean_offset
        self.skinning_weight = p.get_skinning_weight(self.mean_3d, self.scale)
        self.tri_feat = p.extract_tri_feature()
        self.joint_idxs = torch.argmax(self.skinning_weight, 1)
        self.rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None].repeat(smflix.vertex_num_upsampled, 1, 1))
        self.opacity = torch.ones((smflix.vertex_num_upsampled, 1)).float().cuda()
        self.colors = {'rgb': torch.sigmoid(p.rgb), 'seg': torch.sigmoid(p.seg)}
        self.mean_3d_posed = {}

    @torch.no_grad()
    def update_pose(self, param, with_mesh=False):
        if with_mesh:  # fitted SMFLIX mesh (low-res, no Gaussian offsets) for the overlay. world == fitting camera frame
            self.smflix_vert, _ = self.model.get_smflix_outputs(param, self.cam_identity)
        p = self.persona
        mean_offset_offset = p.get_mean_offset_offset(self.tri_feat, param, self.joint_idxs)
        face_offset = (param['expr'][None, None, :] * p.expr_dirs).sum(2) + (param['eyelid'][None, None, :] * p.eyeliddirs).sum(2) # eyelid # Added by Heo
        mean_3d = self.mean_3d + face_offset
        mean_3d_refined = self.mean_3d + mean_offset_offset + face_offset
        skinning_weight_refined = p.get_skinning_weight(mean_3d_refined, self.scale)

        transform_mat_joint = p.get_transform_mat_joint(self.joint_zero_pose, param, jaw_zero_pose=True).view(smflix.joint['num'], 16)
        lbs = lambda xyz, sw: p.lbs(xyz, torch.matmul(sw, transform_mat_joint).view(smflix.vertex_num_upsampled, 4, 4))
        self.mean_3d_posed = {'orig': lbs(mean_3d, self.skinning_weight), 'refined': lbs(mean_3d_refined, skinning_weight_refined)}

    @torch.no_grad()
    def render(self, geometry, color, cam_param, render_shape, bg):
        asset = {'mean_3d': self.mean_3d_posed[geometry], 'opacity': self.opacity, 'scale': self.scale, 'rotation': self.rotation, 'rgb': self.colors[color]}
        return self.renderer(asset, render_shape, cam_param, bg=bg)['img']

    @torch.no_grad()
    def overlay_mesh(self, img, cam_param, render_shape, alpha=0.5):
        # flat gray shading from the face normal's view-direction component, alpha-blended wherever the mesh covers a pixel
        vert = self.smflix_vert @ cam_param['R'].T + cam_param['t']  # world -> camera
        pix_to_face = rasterize(vert[None], smflix.face, {k: v[None] for k, v in cam_param.items()}, render_shape).pix_to_face[0, :, :, 0]  # -1 outside the mesh
        normal = Meshes(vert[None], self.faces[None]).faces_normals_packed()
        shade = 0.3 + 0.7 * normal[pix_to_face.clamp(min=0), 2].abs()
        mask = (pix_to_face >= 0).float()[None]
        return img * (1 - alpha * mask) + shade[None] * alpha * mask

    def bbox(self):
        xyz = self.mean_3d_posed['refined']
        return xyz.min(0)[0].cpu().numpy(), xyz.max(0)[0].cpu().numpy()


# ---------------------------------------------------------------- camera ----------------------------------------------------------------
class OrbitCamera:
    """Orbit camera in the OpenCV convention of the fitting frame (x right, y down, z forward).
    azimuth > 0 moves the camera to the right of the subject, elevation > 0 moves it up."""

    def __init__(self, fovy):
        self.fovy = self.fovy_init = fovy
        self.reset(np.zeros(3, np.float32), 1.0)

    def reset(self, target=None, radius=None):
        if target is not None:
            self.target_init, self.radius_init = np.asarray(target, np.float32), float(radius)
        self.target, self.radius, self.fovy = self.target_init.copy(), self.radius_init, self.fovy_init
        self.azimuth, self.elevation = 0.0, 0.0  # degrees

    def fit(self, bbox_min, bbox_max):
        # look at the body center from a distance that fits the whole body in the vertical field of view
        extent = float(np.max(bbox_max - bbox_min))
        self.reset((bbox_min + bbox_max) / 2, 1.1 * extent / (2 * math.tan(math.radians(self.fovy_init) / 2)))

    def rotation(self):  # camera-to-world
        return rot_y(-math.radians(self.azimuth)) @ rot_x(-math.radians(self.elevation))

    def spin(self, degrees):
        self.azimuth = (self.azimuth + degrees + 180) % 360 - 180

    def orbit(self, dx, dy):
        self.spin(0.3 * dx)
        self.elevation = float(np.clip(self.elevation - 0.3 * dy, -89, 89))

    def zoom(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy):
        self.target += self.rotation() @ np.array([-dx, -dy, 0], np.float32) * (0.001 * self.radius)

    def cam_param(self, render_shape):
        height, width = render_shape
        R_c2w = self.rotation()
        cam_pos = self.target - R_c2w[:, 2] * self.radius
        R = R_c2w.T
        t = -R @ cam_pos
        focal = height / (2 * math.tan(math.radians(self.fovy) / 2))
        return {'R': torch.FloatTensor(R).cuda(), 't': torch.FloatTensor(t).cuda(),
                'focal': torch.FloatTensor([focal, focal]).cuda(), 'princpt': torch.FloatTensor([width / 2, height / 2]).cuda()}


# ---------------------------------------------------------------- viewer ----------------------------------------------------------------
class Viewer:
    def __init__(self, avatar, sources, init_shape, avatar_paths, load_sources, avatar_sources):
        self.avatar, self.sources = avatar, sources
        self.avatar_paths, self.pending_avatar = avatar_paths, None # {name: snapshot .pth} selectable in the panel; a pick is applied in run()
        self.load_sources, self.avatar_sources = load_sources, avatar_sources # the motions of an avatar's own checkpoints, replaced on a switch
        self.set_eye_pose()
        self.cam = OrbitCamera(fovy=25.0)
        self.geometry, self.color, self.bg, self.mesh_overlay = 'refined', 'rgb', 'white', False
        self.playing, self.play_time = False, 0.0
        self.recording, self.record_request = False, False
        self.video_out = None
        self.turntable, self.turntable_speed = False, TURNTABLE_SPEED
        self.pose_dirty = self.cam_dirty = True
        self.pending_resize = None
        self.drag_prev = {}

        # initial window: original image resolution, scaled down if it does not fit the screen
        height, width = init_shape
        if height > MAX_INIT_HEIGHT:
            height, width = MAX_INIT_HEIGHT, int(round(width * MAX_INIT_HEIGHT / height))
        self.render_shape = (height, width)
        self.register_dpg()
        self.set_source(list(sources.keys())[0])

    # ---- state changes ----
    def set_eye_pose(self): # gaze fixed to the captured frame # Added by Heo
        self.eye_pose = {k: v.clone() for k, v in self.sources['captured'][0].items() if k in ('leye_pose', 'reye_pose')} if 'captured' in self.sources else {}

    def set_avatar(self, name):
        # another snapshot under avatars/. the network is shared, so only its weights are loaded and Avatar is rebuilt;
        # the avatar's own motions are replaced, the demo motions kept. the current source, frame and play state are kept if it still exists
        self.pending_avatar = None
        if name == cfg.subject_id:
            return
        if self.recording: # recordings are filed per avatar
            self.toggle_record()
        cfg.subject_id = name
        cfg.model_dir = osp.dirname(self.avatar_paths[name])
        cfg.result_dir = osp.join(osp.dirname(cfg.result_dir), name)
        print('Load checkpoint from ' + self.avatar_paths[name])
        model = self.avatar.model
        model.load_state_dict(rename_body_branch_keys(torch.load(self.avatar_paths[name], map_location='cpu')['network']), strict=False)
        self.avatar = Avatar(model)

        source, frame_idx, playing = self.source, self.frame_idx, self.playing
        new_sources = self.load_sources(name)
        self.sources = {**new_sources, **{k: v for k, v in self.sources.items() if k not in self.avatar_sources}}
        self.avatar_sources = set(new_sources)
        self.set_eye_pose()
        dpg.configure_item('_source', items=list(self.sources.keys()))
        if source in self.sources:
            self.set_source(source)
            self.set_frame(min(frame_idx, len(self.sources[source]) - 1))
            self.playing = playing
        else:
            self.set_source(list(self.sources.keys())[0])
        dpg.set_value('_avatar', name)

    def set_source(self, name):
        if self.recording:
            self.record_request = True
        self.source, self.playing = name, False
        dpg.set_value('_source', name)
        dpg.configure_item('_frame', max_value=len(self.sources[name]) - 1)
        self.set_frame(0)
        self.avatar.update_pose(self.param)
        self.cam.fit(*self.avatar.bbox())
        self.sync_cam_sliders()
        self.cam_dirty = True

    def set_frame(self, frame_idx):
        self.frame_idx = frame_idx
        self.param = {k: v.clone() for k, v in self.sources[self.source][frame_idx].items()}
        self.param.update({k: v.clone() for k, v in self.eye_pose.items()}) # Added by Heo
        dpg.set_value('_frame', frame_idx)
        for name, idx, _ in FACE_SLIDERS:
            dpg.set_value('_face_%s_%d' % (name, idx), self.face_value(name, idx))
        self.pose_dirty = True

    def toggle_record(self):
        self.record_request = False
        self.recording = not self.recording
        if self.recording:
            self.playing = True
            self.set_frame(0) # record the whole motion from its first frame # Added by Heo
            if self.turntable: # start the revolution from the front # Added by Heo
                self.cam.azimuth = 0.0
                self.sync_cam_sliders()
                self.cam_dirty = True
            self.play_time = time.time() # let step() render frame 0 before the first write # Added by Heo
            os.makedirs(cfg.result_dir, exist_ok=True) # output/ is created only when something is recorded
            path = osp.join(cfg.result_dir, 'viewer_%s_%s.mp4' % (self.source, time.strftime('%y%m%d_%H%M%S')))
            self.video_out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), PLAY_FPS, (self.render_shape[1], self.render_shape[0]))
            print('recording to ' + path)
        else:
            self.video_out.release()
            self.video_out = None
        dpg.configure_item('_record', label='Stop rec' if self.recording else 'Record')

    def face_value(self, name, idx, value=None):
        tensor, i = (self.param['body_pose'], NECK_IDX * 3 + idx) if name == 'neck' else (self.param[name], idx)
        if value is not None:
            tensor[i] = value
        return float(tensor[i])

    def sync_cam_sliders(self):
        for name in ('azimuth', 'elevation', 'radius', 'fovy'):
            dpg.set_value('_cam_' + name, getattr(self.cam, name))
            dpg.set_value('_cami_' + name, getattr(self.cam, name)) # Added by Heo

    def reset_camera(self):
        self.turntable = False
        dpg.set_value('_turntable', False)
        self.cam.reset()
        self.sync_cam_sliders()
        self.cam_dirty = True

    # ---- gui ----
    def register_dpg(self):
        height, width = self.render_shape
        dpg.create_context()
        dpg.add_texture_registry(tag='_textures')
        with dpg.window(tag='_image_window', pos=[0, 0], width=width, height=height, no_move=True, no_title_bar=True, no_scrollbar=True, no_resize=True):
            pass
        self.make_texture()
        with dpg.theme() as no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
        dpg.bind_item_theme('_image_window', no_padding)

        with dpg.window(tag='_panel', pos=[width, 0], width=PANEL_WIDTH, height=height, no_move=True, no_title_bar=True, no_resize=True):
            dpg.add_text('', tag='_status')
            with dpg.collapsing_header(label='CAMERA', default_open=True):
                for name, lo, hi in (('azimuth', -180, 180), ('elevation', -89, 89), ('radius', 0.2, 20.0), ('fovy', 5, 120)):
                    with dpg.group(horizontal=True): # the slider for dragging, the box for typing an exact value # Added by Heo
                        dpg.add_slider_float(label='', tag='_cam_' + name, min_value=lo, max_value=hi, width=160, format='%.2f', callback=self.on_cam_slider, user_data=name)
                        dpg.add_input_float(label=name, tag='_cami_' + name, width=90, step=0, format='%.2f', on_enter=True, callback=self.on_cam_slider, user_data=name)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label='Turntable', tag='_turntable', callback=lambda s, a: setattr(self, 'turntable', a))
                    dpg.add_slider_float(label='deg/frame', default_value=TURNTABLE_SPEED, min_value=0.2, max_value=10.0, width=150, format='%.1f', callback=lambda s, a: setattr(self, 'turntable_speed', a))
                dpg.add_button(label='Reset camera', callback=self.reset_camera)
            with dpg.collapsing_header(label='FACE', default_open=True):
                for name, idx, max_value in FACE_SLIDERS:
                    dpg.add_slider_float(label='%s %d' % (name, idx), tag='_face_%s_%d' % (name, idx), min_value=-max_value, max_value=max_value, width=250, format='%.3f', callback=self.on_face_slider, user_data=(name, idx))
                dpg.add_button(label='Reset face', callback=lambda: self.set_frame(self.frame_idx))
            with dpg.collapsing_header(label='AVATAR', default_open=True):
                dpg.add_combo(list(self.avatar_paths.keys()), label='avatar', tag='_avatar', default_value=cfg.subject_id, width=250, callback=lambda s, a: setattr(self, 'pending_avatar', a))
            with dpg.collapsing_header(label='PLAY', default_open=True):
                dpg.add_combo(list(self.sources.keys()), label='source', tag='_source', width=250, callback=lambda s, a: self.set_source(a))
                dpg.add_slider_int(label='frame', tag='_frame', min_value=0, max_value=0, width=250, callback=lambda s, a: self.set_frame(a))
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Play', callback=lambda: setattr(self, 'playing', True))
                    dpg.add_button(label='Stop', callback=lambda: setattr(self, 'playing', False))
                    dpg.add_button(label='Record', tag='_record', callback=lambda: setattr(self, 'record_request', True))
            with dpg.collapsing_header(label='RENDER', default_open=True):
                dpg.add_radio_button(('refined', 'orig'), label='geometry', default_value='refined', horizontal=True, callback=lambda s, a: self.set_option('geometry', a))
                dpg.add_radio_button(('rgb', 'seg'), label='color', default_value='rgb', horizontal=True, callback=lambda s, a: self.set_option('color', a))
                dpg.add_radio_button(('white', 'black'), label='background', default_value='white', horizontal=True, callback=lambda s, a: self.set_option('bg', a))
                dpg.add_checkbox(label='mesh overlay', callback=lambda s, a: (setattr(self, 'mesh_overlay', a), self.set_dirty(pose=True)))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.on_drag)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=self.on_drag)
            dpg.add_mouse_release_handler(callback=lambda: self.drag_prev.clear())
            dpg.add_mouse_wheel_handler(callback=self.on_wheel)

        dpg.create_viewport(title='PERSONA viewer', width=width + PANEL_WIDTH, height=height, resizable=True)
        dpg.set_viewport_resize_callback(self.on_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def make_texture(self):
        # raw textures have a fixed size, so the texture and its image item are rebuilt whenever the render shape changes
        # the texture buffer is a pinned (page-locked) tensor so that the rendered image is copied GPU -> texture memory
        # in one fast DMA transfer. its numpy view is what dearpygui reads; both must stay alive and contiguous
        height, width = self.render_shape
        self.pinned = torch.zeros((height, width, 3), dtype=torch.float32).pin_memory()
        self.buffer = self.pinned.numpy()
        for tag in ('_image', '_texture'):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        dpg.add_raw_texture(width, height, self.buffer, format=dpg.mvFormat_Float_rgb, tag='_texture', parent='_textures')
        dpg.add_image('_texture', tag='_image', parent='_image_window')

    # ---- callbacks ----
    def set_dirty(self, pose=False, cam=False):
        self.pose_dirty |= pose
        self.cam_dirty |= cam

    def set_option(self, name, value):
        setattr(self, name, value)
        self.cam_dirty = True

    def on_cam_slider(self, sender, value, name):
        setattr(self.cam, name, value)
        dpg.set_value('_cam_' + name, value) # slider and box follow each other, whichever was used # Added by Heo
        dpg.set_value('_cami_' + name, value)
        self.cam_dirty = True

    def on_face_slider(self, sender, value, name_idx):
        self.face_value(*name_idx, value=value)
        self.pose_dirty = True

    def on_drag(self, sender, app_data):
        # app_data = [button, dx, dy] accumulated since the drag started, so keep the increment only
        if not dpg.is_item_hovered('_image_window'):
            return
        button, dx, dy = app_data
        prev = self.drag_prev.get(button, (0.0, 0.0))
        self.drag_prev[button] = (dx, dy)
        dx, dy = dx - prev[0], dy - prev[1]
        if button == dpg.mvMouseButton_Left:
            self.cam.orbit(dx, dy)
        else:
            self.cam.pan(dx, dy)
        self.sync_cam_sliders()
        self.cam_dirty = True

    def on_wheel(self, sender, delta):
        if dpg.is_item_hovered('_image_window'):
            self.cam.zoom(delta)
            self.sync_cam_sliders()
            self.cam_dirty = True

    def on_resize(self):
        # debounced: the texture is rebuilt once the window has stopped resizing (see run())
        self.pending_resize = (time.time(), (max(dpg.get_viewport_client_height(), 64), max(dpg.get_viewport_client_width() - PANEL_WIDTH, 64)))

    def apply_resize(self):
        if self.recording:
            self.toggle_record()
        height, width = self.render_shape = self.pending_resize[1]
        self.pending_resize = None
        dpg.configure_item('_image_window', width=width, height=height)
        dpg.configure_item('_panel', pos=[width, 0], height=height)
        self.make_texture()
        self.cam_dirty = True

    # ---- main loop ----
    def step(self):
        if not (self.pose_dirty or self.cam_dirty):
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if self.pose_dirty:
            self.avatar.update_pose(self.param, with_mesh=self.mesh_overlay)
        bg = torch.ones(3).float().cuda() * (self.bg == 'white')
        cam_param = self.cam.cam_param(self.render_shape)
        img = self.avatar.render(self.geometry, self.color, cam_param, self.render_shape, bg)
        if self.mesh_overlay:
            img = self.avatar.overlay_mesh(img, cam_param, self.render_shape)
        self.pinned.copy_(img.permute(1, 2, 0).clamp(0, 1))  # GPU -> pinned host memory, no intermediate numpy copy
        dpg.set_value('_texture', self.buffer)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        dpg.set_value('_status', '%s  frame %d  |  %dx%d  |  %.1f ms (%d FPS)' % (self.source, self.frame_idx, self.render_shape[1], self.render_shape[0], ms, 1000 / ms))
        self.pose_dirty = self.cam_dirty = False

    def run(self):
        while dpg.is_dearpygui_running():
            if self.pending_resize is not None and time.time() - self.pending_resize[0] > 0.1:
                self.apply_resize()
            if self.pending_avatar is not None: # applied here, not in the combo callback, so it never runs in the middle of a render
                self.set_avatar(self.pending_avatar)
            if self.record_request:
                self.toggle_record()
            new_frame = False
            if self.playing and time.time() - self.play_time >= 1.0 / PLAY_FPS:
                new_frame = True
                self.play_time = time.time()
                if self.recording:
                    self.video_out.write(cv2.cvtColor((self.buffer * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if len(self.sources[self.source]) > 1 and self.frame_idx + 1 == len(self.sources[self.source]):
                        self.toggle_record()
                self.set_frame((self.frame_idx + 1) % len(self.sources[self.source]))
            if self.turntable and (new_frame or not self.recording):
                self.cam.spin(self.turntable_speed)
                self.sync_cam_sliders()
                self.cam_dirty = True
            self.step()
            dpg.render_dearpygui_frame()
        if self.recording:
            self.video_out.release()
        dpg.destroy_context()


# ---------------------------------------------------------------- eval ----------------------------------------------------------------
@torch.no_grad()
def run_eval(avatar, frames, save_path): # Added by Heo
    bg = torch.ones(3).float().cuda() # white, as Eval_Avatar/main/eval.py fills outside the GT mask
    for image_id in tqdm(sorted(frames)):
        param, cam_param, render_shape = frames[image_id]
        avatar.update_pose(param)
        img = avatar.render('refined', 'rgb', cam_param, render_shape, bg) # (3,H,W) RGB 0..1, the '_refined' output of test.py
        img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).round().astype(np.uint8)
        cv2.imwrite(osp.join(save_path, '%d.png' % image_id), img)


# ---------------------------------------------------------------- main ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id', default=None) # default: the first avatar under avatars/ with this epoch's snapshot
    parser.add_argument('--test_epoch', type=str, dest='test_epoch', default='4')
    parser.add_argument('--head_no_pose_offset', dest='head_no_pose_offset', action='store_true') # no pose-dependent offset on the whole head (cfg.head_no_pose_offset) # Added by Oh
    parser.add_argument('--motion_path', type=str, dest='motion_path', default=None)
    parser.add_argument('--motion_ckpt', type=str, dest='motion_ckpt', nargs='*', default=[]) # extra checkpoints whose splits are added as motion sources # Added by Heo
    parser.add_argument('--motion_root', type=str, dest='motion_root', default=osp.join('..', 'demo')) # every <motion_root>/*/smplx_init is added as a motion source # Added by Heo
    parser.add_argument('--eval_path', type=str, dest='eval_path', default=None) # smflix_optimized.json. renders its frames with its cameras into save_path, without the window # Added by Heo
    parser.add_argument('--save_path', type=str, dest='save_path', default=None) # the rendered images, {image_id}.png at the original resolution # Added by Heo
    args = parser.parse_args()
    if args.subject_id is None:
        snapshots = sorted(glob(osp.join(cfg.root_dir, 'avatars', '*', 'snapshot_%d.pth' % int(args.test_epoch))))
        assert snapshots, 'No avatars/*/snapshot_%s.pth' % args.test_epoch
        args.subject_id = osp.basename(osp.dirname(snapshots[0]))
    assert not args.eval_path or args.save_path, 'Please set save_path.'
    return args


def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    if args.head_no_pose_offset: # Added by Oh
        cfg.head_no_pose_offset = True

    # load the trained avatar without a dataset (same as animate.py). identity and textures come from the checkpoint
    tester = Tester(args.test_epoch)
    smflix.set_id_info(None, None, None, None)
    smflix.set_texture(None, None, None)
    tester._make_model()
    avatar = Avatar(tester.model.module)

    # eval mode: render the frames of smflix_optimized.json with their cameras, without the window # Added by Heo
    if args.eval_path:
        frames = load_eval_params(args.eval_path)
        os.makedirs(args.save_path, exist_ok=True)
        image_id = sorted(frames)[0]
        _, cam_param, render_shape = frames[image_id]
        print('eval: %d frames from %s' % (len(frames), args.eval_path))
        print('eval: frame %d  render %dx%d (h x w)  focal %s  princpt %s' % (image_id, render_shape[0], render_shape[1], cam_param['focal'].tolist(), cam_param['princpt'].tolist()))
        print('eval: save to ' + args.save_path)
        run_eval(avatar, frames, args.save_path)
        return

    # motion sources and the initial window size (original image resolution)
    # an avatar's own motions (its checkpoint splits). with --motion_path there are none, also after an avatar switch
    load_sources = (lambda name: {}) if args.motion_path else (lambda name: load_avatar_sources(name, int(args.test_epoch), args.motion_ckpt))
    if args.motion_path:
        sources, img_path = load_motion_params(args.motion_path)
        avatar_sources = set()
    else:
        sources = load_sources(cfg.subject_id)
        avatar_sources = set(sources)
        img_path = osp.join('..', 'data', 'subjects', cfg.subject_id, 'captured', 'images', '0.png')
    for motion_path in sorted(glob(osp.join(args.motion_root, '*', 'smplx_init'))): # demo motions from run_smflix.py # Added by Heo
        motion_path = osp.dirname(motion_path)
        sources.setdefault(osp.basename(motion_path), load_motion_params(motion_path)[0]['motion'])
    img = cv2.imread(img_path) if osp.isfile(img_path) else None
    init_shape = img.shape[:2] if img is not None else (1024, 576) # no subject folder for backup/...: default window # Added by Heo

    # every avatar with this epoch's snapshot under avatars/ is selectable in the panel
    avatar_paths = {osp.basename(osp.dirname(p)): p for p in sorted(glob(osp.join(osp.dirname(cfg.model_dir), '*', 'snapshot_%d.pth' % int(args.test_epoch))))}
    Viewer(avatar, sources, init_shape, avatar_paths, load_sources, avatar_sources).run()


if __name__ == '__main__':
    main()
