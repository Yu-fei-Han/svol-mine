import argparse
import GPUtil
import os
import gc
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets.data_io import read_pfm, save_pfm
import volsdf.utils.general as utils
import volsdf.utils.plots as plt
from volsdf.datasets.scene_dataset import get_trains_ids, get_eval_ids
from helpers.help import logger

torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

def evaluate(**kwargs):
    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    root = './'
    expname = kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)
    else:
        scan_id = conf.get_string('dataset.object', default='')

    if kwargs['ckpt_dir'] == '' and kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join(root, kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join(root, kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            # self.timestamp = sorted(timestamps)[-1]
            timestamp = None
            for t in sorted(timestamps):
                if os.path.exists(os.path.join(root, kwargs['exps_folder_name'], expname, t, 'checkpoints',
                                               'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    timestamp = t
            if timestamp is None:
                print('NO GOOD TIMSTAMP')
                exit()
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join(root, evals_folder_name))
    expdir = os.path.join(root, exps_folder_name, expname)
    evaldir = os.path.join(root, evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    dataset_conf['num_views'] = -1 # all images
    dataset_conf['data_dir_root'] = opt.data_dir_root
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)
    print(len(eval_dataset))
    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()

    if opt.eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      num_workers=0,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res
        logger.info(f'img res = {img_res}')

    if kwargs['ckpt_dir'] != '':
        old_checkpnts_dir = os.path.join(root, kwargs['ckpt_dir'], 'checkpoints')
    else:
        old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    logger.info(f'load model from: {old_checkpnts_dir}')
    if opt.result_from != 'None':
        epoch = 0
        # use the latest epoch's rendering results
        for renderdir in os.listdir(evaldir):
            if renderdir.startswith('rendering_'):
                epoch = max(epoch, int(renderdir.replace('rendering_', '')))
    else:
        saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        model.load_state_dict(saved_model_state["model_state_dict"])
        epoch = saved_model_state['epoch']

    ####################################################################################################################
    model.cuda()
    model.eval()

    

    if opt.eval_rendering:
        images_dir = '{0}/rendering_{1}'.format(evaldir, epoch)
        logger.info(f"rendered images dir: {images_dir}")
        utils.mkdir_ifnotexists(images_dir)
        os.makedirs(os.path.join(images_dir, 'depth_est'), exist_ok=True)

        if 'dtu' in opt.conf:
            train_idx = get_trains_ids('DTU', scan=None, num_views=48)
        elif 'bmvs' in opt.conf:
            test_idx = get_eval_ids('BlendedMVS', scan_id)
            train_idx = get_trains_ids('BlendedMVS', f'scan{scan_id}', num_views=3)
            assert test_idx == [i for i in test_idx if i not in train_idx]
        else:
            raise NotImplementedError

        test_idx = list(range(len(eval_dataset)))
        logger.info(f"{len(test_idx)} images (including train)")

        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            if indices not in test_idx: continue

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

        
            split = utils.split_input(model_input, total_pixels, n_pixels=opt.split_n_pixels)
            res = []
            for s in tqdm(split, ncols=60):
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                    'normal_map': out['normal_map'].detach(),
                    'depth_values': out['depth_values'].detach(),
                    'weights': out['weights'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)


            ## Normal
            normal_eval = model_outputs['normal_map']
            normal_map = normal_eval.reshape(batch_size, total_pixels, 3)
            normal_map = torch.nn.functional.normalize(normal_map,dim=-1)

            R = model_input['pose'][:,:3,:3]
            R = torch.linalg.inv(R)
            normal_map = torch.matmul(R,normal_map[...,None]).squeeze(-1)
            normal = torch.zeros_like(normal_map)
            normal[...,0] = normal_map[...,0]
            normal[...,1] = -normal_map[...,1]
            normal[...,2] = -normal_map[...,2]
            normal_map = (normal + 1.) / 2.


            normal_map = plt.lin2img(normal_map, img_res).detach().cpu().numpy()[0]
            normal_map = normal_map.transpose(1, 2, 0)
            cv2.imwrite('{0}/normal_{1}.png'.format(images_dir, '%03d' % indices[0]),(normal_map[...,::-1]*65535).astype(np.uint16))

            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    ## prepare data
    # python eval_vsdf.py --conf dtu  --data_dir_root data_s_volsdf --scan_ids 106 --eval_rendering --gpu 0 --expname ours --exps_folder exps_vsdf --evals_folder exps_result # --eval_mesh
    # python eval_vsdf.py --conf bmvs --data_dir_root data_s_volsdf --scan_ids 4   --eval_mesh --eval_rendering --gpu 1 --expname ours --exps_folder exps_vsdf --evals_folder exps_result # --eval_mesh

    ## evaluate
    # python eval_vsdf.py --conf dtu  --data_dir_root data_s_volsdf --eval_rendering --gpu 0 --expname ours --exps_folder exps_vsdf --evals_folder exps_result --result_from blend
    # python eval_vsdf.py --conf bmvs --data_dir_root data_s_volsdf --eval_rendering --gpu 1 --expname ours --exps_folder exps_vsdf --evals_folder exps_result --result_from blend

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='dtu')
    parser.add_argument('--data_dir_root', type=str, default='test_data', help='GT data dir')
    parser.add_argument('--eval_mesh', default=False, action="store_true", help='extract mesh via marching cube')
    parser.add_argument('--eval_rendering', default=True, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--result_from', default='None', type=str, choices=['None', 'default', 'blend'])
    parser.add_argument('--expname', type=str, default='ours', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps_vsdf', help='The experiments folder name for train.')
    parser.add_argument('--evals_folder', type=str, default='exps_result', help='The evaluation folder name (a new folder).')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str, help='The trained model checkpoint to test')
    parser.add_argument('--ckpt_dir', default='',type=str)
    parser.add_argument('--scan_ids', nargs='+', type=int, default=[141], help='e.g. --scan_ids 12 34 56')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube, set as 400 if not enough GPU')
    parser.add_argument('--split_n_pixels', default=2048, type=int)
    opt = parser.parse_args()

    # configs
    opt.conf = f'./config/confs/{opt.conf}.conf'
    opt.eval_rendering = (opt.result_from != 'None') or opt.eval_rendering
    if opt.scan_ids is None:
        if 'dtu' in opt.conf:
            opt.scan_ids = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
        elif 'bmvs' in opt.conf:
            opt.scan_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            raise NotImplementedError
    assert opt.result_from in ['None', 'default', 'blend']
    assert os.path.exists(opt.exps_folder)
    logger.warning(f'result folder = {opt.evals_folder}')
    logger.warning(f'volsdf ckpt folder = {opt.exps_folder}/{opt.expname}')
    logger.info(f'scan_ids = {opt.scan_ids}')

    # GPU
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.9, maxMemory=0.9, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    # eval

    for scan_id in opt.scan_ids:
        logger.info(f'scan_id = {scan_id}')
        evaluate(
                conf=opt.conf,
                expname=opt.expname,
                exps_folder_name=opt.exps_folder,
                evals_folder_name=opt.evals_folder,
                timestamp=opt.timestamp,
                checkpoint=opt.checkpoint,
                scan_id=int(scan_id),
                resolution=opt.resolution,
                ckpt_dir=opt.ckpt_dir,
                )

