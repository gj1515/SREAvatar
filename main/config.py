import os
import os.path as osp
import sys


class Config:
    ## shape
    smflix_uvmap_shape = (1024, 1024)  # height, width
    triplane_shape_3d = (2, 2, 2)  # meter
    triplane_shape = (32, 128, 128)  # feat_dim, height, width
    face_patch_shape = (256, 256)  # height, width

    ## train
    lr = 1e-3
    smflix_param_lr = 1e-3
    boundary_thr = 0.02
    end_epoch = 5
    vis_every_itr = 1  # [side_vis] every N iterations of the body pass: img_refined panel + side view under debug_vis/itr/. 0: off # Added by Heo
    img_mask_gate = False  # [gate] multiply the image losses (full image + part crops) by the GT mask: no colour loss where GT is background. False: as before # Added by Heo
    img_mask_erode = 0  # [gate] px the GT mask is eroded by before gating (0: no erosion) # Added by Heo
    img_kernel_ratio = 0.015
    patch_kernel_ratio = 0.06

    ## loss functions
    rgb_loss_weight = 0.8
    ssim_loss_weight = 0.2
    lpips_loss_weight = 1.0
    depth_loss_weight = 0.01
    lap_rgb_weight = 0.1  # [laprgb] laplacian colour smoothing on the body kinds (head_syn keeps 0.1 on the head rows). 0.1 before # Added by Heo
    fit_no_geo = False  # [no_geo] train.py --fit_pose_to_test --no_geo: image losses only, geo / geo_refined (mask, depth, normal, seg) dropped # Added by Heo
    head_no_pose_offset = True  # True: no mean_offset_offset (pose-dependent MLP offset) on the whole head (is_head: Head/Eye/Jaw/Neck Gaussians), not only the face expression region and eyes. set by --head_no_pose_offset of the inference scripts # Added by Oh
    img_boundary_weight = 1.0  # [bdw] img_boundary / img_boundary_refined (the boundary band colour pulled to the rgb_aux render, captured anchors). 1 before # Added by Heo


    ## train_sr (avatar/main/train_sr.py). [train_sr]
    # one mixed epoch of two passes, continuing from snapshot_{sr_init_epoch} in the same model_dir:sr_boundary_generated
    #   head pass (Model.forward_head): head_splits face frames + the open / closed captured face-patch anchors
    #   body pass (Model.forward_body): sr_body_splits SCAIL frames + sr_real_splits full images with their SR part crops
    # one Adam for both passes; each pass updates only its own rows (PERSONA.row_mask_head / row_mask_body), so the head pass
    # cannot drag the shoulders and the body pass cannot redo the head. triplane and the pose-dependent MLP are trained by both
    train_sr = False
    sr_init_epoch = None
    sr_epoch = 5  # snapshot_{init+1} .. snapshot_{init+sr_epoch} go to the same model_dir
    sr_resume = False  # [resume] train_sr.py --resume: continue snapshot_{sr_init_epoch} as saved (per-frame params, identity, Adam state, its last lr) instead of fine-tuning from it # Added by Heo
    sr_resume_lr = 'config'  # [resume] 'saved': the snapshot's last lr, kept constant. 'config': Adam state from the snapshot, lr from sr_lr with the set_lr schedule # Added by Heo
    sr_lr = 1e-3  # avatar base lr (scale/rgb/rgb_aux/seg use x10) and the per-frame param lr. the body branch's value. [exp1] 5e-4 for the face-only fine-tuning; scratch runs use 1e-3 # Added by Heo
    sr_lr_warmup_itr = 0  # linear lr ramp over the first itrs. 0 = off (measured: 5e-4 without it bumps the anchor loss +33% for ~100 itrs, then recovers)
    sr_fixed_order = False  # [fixed_order] no shuffle: every epoch runs all body frames, then all body anchors, then all face frames, then all face anchors. False: shuffled # Added by Heo
    sr_train_global = True  # train triplane and the offset MLP in train_sr. False: the snapshot's are kept (to_give: they are global, so a pass cannot own them). [exp1-3] False; the other runs use True # Added by Heo
    # head pass
    head_splits = ('generated_face_0',)  # LivePortrait face crops of the neutral head turns, with the HyPlaneHead orbit frames merged in
    head_frame_interval = 1  # subsampling of the head_splits frames. [exp1-2] 1 (to_give's value); the scratch runs so far used 2, back to 1 for subjects generated at half the frames (341 -> 171) # Added by Heo
    head_anchor_ratio = 1.0  # captured face-patch anchor iterations per face frame. 0 disables the anchor
    head_anchor_closed_frac = 0.5  # share of those anchors that use the closed-eye patch (captured/face_patches_closed, preprocess/tools/run_closed_anchor.py)
    head_geo_weight = 0.1  # depth / normal / seg weight of the face frames (same as captured)
    # body pass
    sr_body_splits = ('generated_3', 'generated_7')  # SCAIL-2 body frames (preprocess/tools/run_generated_scail.py). [exp1] (): no body frames and so no body anchors, the head pass only; the other runs use ('generated_3', 'generated_7') # Added by Heo
    sr_body_frame_interval = 1
    sr_real_splits = ('captured', 'captured_sr')  # body anchors: the full real images with their part crops, together 1:1 with the body frames like train.py's captured. [exp1] (): none ('captured' is still loaded for the face-patch anchors); the other runs use ('captured', 'captured_sr') # Added by Heo
    sr_body_anchor_ratio = 1.0  # real anchor iterations per body frame, split evenly over sr_real_splits
    sr_body_loss_weight = 1.0  # every loss term of the body pass is multiplied by this. 1: as before # Added by Heo
    sr_anchor_balance = True  # False: every anchor (head open / closed, each real split) once an epoch instead of 1:1 with the synthetic frames # Added by Heo
    sr_boundary_generated = True  # body branch version2: the boundary down-weighting (x0.1) of the image loss on generated frames too, not only captured. [step5] False: the body frames get 1 as in train.py; the runs up to step4 used True # Added by Heo
    sr_body_weight = 'none'  # extra per-pixel weight of the image loss on the body pass. 'view': n_z of the rendered normal (body frames and real anchors). 'seg_map': the sr_weight_map_dir maps (body frames only). 'none' # Added by Heo
    sr_weight_map_dir = 'weight_maps_seg_gauss_img'  # per-frame weight map of the synthetic body frames, under <split>/smplx_optimized (preprocess/main/make_seg_weight_map.py). [seggauss] 'weight_maps_seg_gauss_img': mean / variance; [segmaj] 'weight_maps_seg_img': majority # Added by Heo
    sr_body_face = False  # [train_sr] face and eye terms on the body pass (the face patch losses of its real anchors, face_geo, mask_eye).
                          # False: the head pass owns the face. their rows are frozen here by the row mask anyway, so the losses and their 9 renders were pure cost
    sr_vis_interval = 25  # frame interval of the sanity renders (train_sr.py save_vis)
    # part crops (preprocess/tools/crop_seg_parts.py; SR'd for captured_sr by run_captured.py), trained like the face patch (Model.patch_losses)
    parts_patch_dir = {'captured': 'parts_crop', 'captured_sr': 'parts_crop_sr'}  # <split>/<dir>/<part>/{images,bbox}/<frame>.*
    parts_patch_names = ('ubody', 'lbody', 'l_arm', 'r_arm', 'l_upper_arm', 'r_upper_arm', 'l_lower_arm', 'r_lower_arm', 'l_hand', 'r_hand', 'upper_leg', 'lower_leg', 'foot')  # head left out: the face patch covers it
    parts_patch_weight = 0.1  # every part (the face patch losses use 0.1). [step3_partsw01] 0.1; the runs so far used 0.5 # Added by Heo
    parts_patch_loss = 'img'  # 'img': image + mask losses. 'face': also the _boundary / _geo terms (3 more renders + 1 LPIPS per part)
    body_patch_dirs = {}  # 'part_<name>' -> '<name>', filled by set_args(parts=...). {} = no part patches
    body_patch_weight = {}

    ## others
    num_thread = 2 # each DataLoader worker holds a 0.59GB CUDA context (smflix import); 16 workers took 8.4GB of the 32GB and spilled training to system RAM # Added by Heo
    num_gpus = 1
    batch_size = 1  # Gaussian splatting renderer only supports batch_size==1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(root_dir, 'avatars')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join('..', 'common', 'utils', 'human_model_files')
    sw_path = osp.join('..', 'tools', 'diffused_skinning_weights')

    def set_args(self, subject_id, fit_pose_to_test=False, train_sr=False, sr_init_epoch=None, parts=None):
        self.subject_id = subject_id
        self.fit_pose_to_test = fit_pose_to_test
        # [train_sr] continue from snapshot_{sr_init_epoch}; snapshot_{init+1}.. go to the same model_dir.
        # parts: 'all', 'none' or a comma-separated subset of parts_patch_names -> body_patch_dirs / body_patch_weight
        self.train_sr = train_sr
        self.sr_init_epoch = sr_init_epoch
        if self.train_sr:
            assert not self.fit_pose_to_test, 'train_sr is exclusive'
            # Added by Heo
            self.end_epoch = self.sr_epoch if self.sr_init_epoch is None else self.sr_init_epoch + 1 + self.sr_epoch
            self.lr = self.sr_lr  # get_optimizable_params / SMFLIXParamDict read cfg.lr and cfg.smflix_param_lr
            self.smflix_param_lr = self.sr_lr
            names = list(self.parts_patch_names) if parts in (None, 'all') else [x for x in parts.split(',') if x and x != 'none']
            assert all(n in self.parts_patch_names for n in names), 'unknown part in ' + str(parts) + ', expected ' + ','.join(self.parts_patch_names)
            self.body_patch_dirs = {'part_' + n: n for n in names}
            self.body_patch_weight = {'part_' + n: self.parts_patch_weight for n in names}
        if self.fit_pose_to_test:
            self.model_dir = osp.join(self.model_dir, subject_id + '_fit_pose_to_test')
            self.result_dir = osp.join(self.result_dir, subject_id + '_fit_pose_to_test')
        else:
            self.model_dir = osp.join(self.model_dir, subject_id)
            self.result_dir = osp.join(self.result_dir, subject_id)
            self.log_dir = osp.join(self.log_dir, subject_id) # per-subject logs # Added by Heo


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath

add_pypath(osp.join(cfg.data_dir))
