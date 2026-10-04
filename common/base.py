import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
import numpy as np
from config import cfg
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_model

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

def rename_body_branch_keys(network):
    # [train_sr] checkpoints of the body-training branch name the per-frame dict smplx_param_dict.smplx_params (SMPL-X-era
    # name); ours is smflix_param_dict.smflix_params. same layout otherwise (split.frame.name, 6D poses), so rename in place
    old, new = 'smplx_param_dict.smplx_params.', 'smflix_param_dict.smflix_params.'
    return {(new + k[len(old):] if k.startswith(old) else k): v for k, v in network.items()}


def get_finetune_avatar_state(network):
    # Additional training restores avatar assets, not the old fitting results.
    # Dataset initialization supplies identity from captured and annotations from each current split.
    identity_keys = {'persona.shape_param', 'persona.face_offset', 'persona.joint_offset', 'persona.betas_head'}
    return {
        key: value for key, value in network.items()
        if key not in identity_keys and not key.startswith(('smflix_param_dict.', 'smplx_param_dict.'))
    }


class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, optimizable_params):
        optimizer = torch.optim.Adam(optimizable_params, lr=0.0, eps=1e-15)
        return optimizer
 
    def set_lr(self, cur_itr, tot_itr):
        if cfg.train_sr and cfg.sr_resume and cfg.sr_resume_lr == 'saved': # [resume] # Added by Heo
            return
        if cfg.train_sr:
            # [train_sr] stateless schedule from the stored base_lr: linear ramp over the first sr_lr_warmup_itr itrs
            # (a fresh Adam), then /10 at 75% like the original schedule
            factor = min(1.0, (cur_itr + 1) / cfg.sr_lr_warmup_itr) if cfg.sr_lr_warmup_itr > 0 else 1.0
            if cur_itr >= int(0.75 * tot_itr):
                factor *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['base_lr'] * factor
            return
        if cur_itr == int(0.75 * tot_itr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 10
   
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        from dataset import Dataset # data/ is not shipped with the viewer
        trainset_loader = Dataset(transforms.ToTensor(), 'train')
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=not (cfg.train_sr and cfg.sr_fixed_order), num_workers=cfg.num_thread, pin_memory=True) # [fixed_order] # Added by Heo
        self.smflix_params = trainset_loader.smflix_params

    def _make_model(self, epoch=None):
        model = get_model(self.smflix_params)
        model = DataParallel(model).cuda()
        if cfg.fit_pose_to_test:
            ckpt = self.load_model()
            model.module.load_state_dict(ckpt['network'], strict=False)
            start_epoch = ckpt['epoch'] + 1
        elif cfg.train_sr and cfg.sr_init_epoch is None:
            start_epoch = 0 # [train_sr] from scratch: no snapshot, every parameter trained (model.py) # Added by Heo
        elif cfg.train_sr and cfg.sr_resume: # [resume] # Added by Heo
            ckpt = self.load_model(cfg.sr_init_epoch)
            missing, unexpected = model.module.load_state_dict(ckpt['network'], strict=False)
            missing = [k for k in missing if 'smflix_layer' not in k and 'lpips' not in k]
            skipped = [k for k in unexpected if k.startswith('smflix_param_dict.')] # the splits this dataset does not load # Added by Heo
            unexpected = [k for k in unexpected if k not in skipped]
            assert not missing and not unexpected, 'snapshot does not match this dataset: missing %s, unexpected %s' % (missing[:5], unexpected[:5])
            self.logger.info('Resume: network loaded as saved (%d entries, %d of splits not loaded skipped)' % (len(ckpt['network']), len(skipped)))
            start_epoch = ckpt['epoch'] + 1
        elif cfg.train_sr:
            # [train_sr] initialize avatar assets from the snapshot, but keep CURRENT dataset fitting.
            # This starts additional training; it is not an exact resume of the snapshot's per-frame optimization.
            ckpt = self.load_model(cfg.sr_init_epoch)
            assert 'persona.betas_head' in ckpt['network'], 'the checkpoint was not trained with the SMFLIX avatar (no persona.betas_head)'
            network = get_finetune_avatar_state(ckpt['network'])
            model.module.load_state_dict(network, strict=False)
            model.module.smflix_param_dict.sync_closed_anchor()  # current captured annotation; keep the fitted closed eyelid
            self.logger.info('Fine-tuning load: avatar assets from snapshot; identity from current captured JSON; '
                             'pose/expression/eyelid/translation from current per-split annotations '
                             '(ignored %d checkpoint identity/annotation entries).' % (len(ckpt['network']) - len(network)))
            start_epoch = ckpt['epoch'] + 1
        else:
            start_epoch = 0
        optimizer = self.get_optimizer(model.module.optimizable_params)
        if cfg.train_sr:
            # [train_sr] remember the base lr of every group; set_lr rescales from it each itr
            for param_group in optimizer.param_groups:
                param_group['base_lr'] = param_group['lr']
        if cfg.train_sr and cfg.sr_resume: # [resume] # Added by Heo
            # matched by group name: the groups of this run can be a subset of the snapshot's (--head_only, sr_train_global False) # Added by Heo
            saved = {g['name']: g for g in ckpt['optimizer']['param_groups']}
            missing = [g['name'] for g in optimizer.param_groups if g['name'] not in saved]
            assert not missing, 'optimizer groups not in the snapshot: %s' % missing[:5]
            for group in optimizer.param_groups:
                if cfg.sr_resume_lr == 'saved':
                    group['lr'] = saved[group['name']]['lr']
                for param, idx in zip(group['params'], saved[group['name']]['params']):
                    state = ckpt['optimizer']['state'][idx]
                    assert state['exp_avg'].shape == param.shape, 'optimizer state of %s does not match' % group['name']
                    optimizer.state[param] = {k: (v if k == 'step' else v.to(param.device)) for k, v in state.items()}
            self.logger.info('Resume: optimizer loaded (%d groups, %d of the snapshot not used), lr %s from %s' % (len(optimizer.param_groups),
                             len(saved) - len(optimizer.param_groups), cfg.sr_resume_lr,
                             ', '.join('%s %g' % (g['name'], g['lr']) for g in optimizer.param_groups[:5])))

        model.train()
        for module in model.module.eval_modules:
            module.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth'.format(str(epoch)))

        # exclude some keys when saving the checkpoint
        exclude_keys = []
        for k in state['network'].keys():
            if ('smflix_layer' in k) or ('lpips' in k):
                exclude_keys.append(k)
        for k in exclude_keys:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, epoch=None):
        if epoch is None:  # latest snapshot ([train_sr] passes an explicit epoch)
            model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth'))
            epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth')]) for file_name in model_file_list])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(epoch) + '.pth')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')
        ckpt['network'] = rename_body_branch_keys(ckpt['network'])  # [train_sr]
        return ckpt

class Tester(Base):
    def __init__(self, test_epoch):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.test_epoch = int(test_epoch)
        self.smflix_params = None

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        from dataset import Dataset # data/ is not shipped with the viewer
        testset_loader = Dataset(transforms.ToTensor(), 'test')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator
        self.smflix_params = testset_loader.smflix_params

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        ckpt['network'] = rename_body_branch_keys(ckpt['network'])  # [train_sr]
        # checkpoints trained with the SMPL-X avatar lack the SMFLIX identity buffer. strict=False would
        # silently load them half-way, so refuse them explicitly.
        if getattr(self, 'allow_smplx_ckpt', False) and 'persona.betas_head' not in ckpt['network']:
            # [official_baseline] an SMPL-X avatar: SMFLIX has the same template, faces and skinning, and betas_head = 0
            # keeps the SMPL-X head, so only this buffer is missing. viewers only; training keeps the check below # Added by Heo
            self.logger.info('SMPL-X checkpoint: persona.betas_head set to zeros')
            ckpt['network']['persona.betas_head'] = torch.zeros(300) # smflix.betas_head_dim
            # the two do not share shapedirs (up to 8.6mm apart), so this avatar's shape_param only reproduces its
            # mesh with SMPL-X's own basis. # Added by Heo
            from utils.smflix import smflix
            smplx_shapedirs = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_NEUTRAL.npz'))['shapedirs']
            smflix.layer.shapedirs.copy_(torch.as_tensor(smplx_shapedirs[:, :, :smflix.shape_param_dim]).to(smflix.layer.shapedirs))
            self.logger.info('SMPL-X checkpoint: shapedirs replaced with SMPLX_NEUTRAL')
        assert 'persona.betas_head' in ckpt['network'], model_path + ' was not trained with the SMFLIX avatar (no persona.betas_head). Retrain with the current code.'

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model()
        model = DataParallel(model).cuda()
        model.module.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

