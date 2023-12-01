import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_h36m_fetch_all_test import H36MDataset
from models_def_test import DepthAngleEstimator, DepthAngleEstimator_transformer
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics as m
from utils.metrics_batch import Metrics as mb
from defs import *
from utils.helpers import *

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import argparse
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
parser.add_argument("-n", "--num_bases", help="number of PCA bases", type=int, default=26)
parser.add_argument("-b", "--bl", help="bone lengths", type=float, default=50.0)
parser.add_argument("-t", "--translation", help="camera translation", type=float, default=10.0)
parser.add_argument("-r", "--rep2d", help="2d reprojection", type=float, default=1.0)
parser.add_argument("-o", "--rot3d", help="3d reconstruction", type=float, default=1.0)
parser.add_argument("-v", "--velocity", help="velocity", type=float, default=1.0)

args = parser.parse_args()
num_bases = args.num_bases

# Configuration
config = {
    "learning_rate": 0.0002,
    "BATCH_SIZE": 256,
    "N_epochs": 100,
    "use_elevation": True,
    "weight_bl": args.bl,
    "weight_2d": args.rep2d,
    "weight_3d": args.rot3d,
    "weight_velocity": args.velocity,
    "depth": args.translation,
    "use_gt": True,
    "num_joints": 17,
    "num_bases": num_bases,
    "trainfile": data_folder + 'H36M/h36m_train_detections.pkl',
    "testfile": data_folder + 'H36M/h36m_test_detections.pkl',
    #"testfile": '/netscratch/satti/mpi_inf_3dhp/3dhp_test_detections.pkl',
    "checkpoints": project_folder + 'checkpoints/',
    "receptive_field": 17,
    "model": "poseformer",
    #"checkpoint": "/netscratch/satti/experiments/Elepose-Poseformer-correct-dataset/checkpoints-WD5/epoch_15.bin"
    "checkpoint": '/netscratch/satti/experiments/Elepose-Poseformer-correct-dataset/checkpoints-final/epoch_37.bin'
}

print("checkpoint: ", config['checkpoint'])

# Load pretrained INN
inn_2d_1 = Ff.SequenceINN(num_bases)
for k in range(8):
    inn_2d_1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

inn_2d_1.load_state_dict(torch.load(
    project_folder + 'models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))

for param in inn_2d_1.parameters():
    param.requires_grad = False


# Defining the architecture
if (config['model'] == 'elepose'):
    depth_estimator = DepthAngleEstimator(use_batchnorm=False, num_joints=config['num_joints'])
elif (config['model'] == 'poseformer'):
    depth_estimator = DepthAngleEstimator_transformer(num_frame=9, num_joints=config['num_joints'], in_chans=2, embed_dim_ratio=32, depth=4,
                                    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.0)

if torch.cuda.is_available():
    print("CUDA Available")
    depth_estimator = nn.DataParallel(depth_estimator)
    depth_estimator = depth_estimator.cuda()
    inn_2d_1 = inn_2d_1.cuda()


if config["use_gt"]:
    test_dataset = H36MDataset(config["testfile"], get_PCA=False, normalize_func=normalize_head_test, stage="test", receptive_field=9)
else:
    test_dataset = H36MDataset(config["testfile"], get_PCA=False, normalize_func=normalize_head_test, stage="test", receptive_field=9)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, num_workers=16)

bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                    0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()

checkpoint = torch.load(config['checkpoint'])
depth_estimator.load_state_dict(checkpoint['model_pos'])

results = {}
with torch.no_grad():
    #Evaluation mode
    depth_estimator.eval()

    mpjpe_loss_per_epoch = []
    pmpjpe_loss_per_epoch = []
    pck_loss_per_epoch = []
    auc_loss_per_epoch = []

    for val_batch in test_loader:

        test_poses_2dgt_normalized = val_batch['p2d_gt']     
        test_3dgt_normalized = val_batch['poses_3d']

        if (config['model'] == 'elepose'):
            test_poses_2dgt_normalized = test_poses_2dgt_normalized[:, 4, :]           

        if torch.cuda.is_available():
            test_poses_2dgt_normalized = test_poses_2dgt_normalized.cuda()
            test_3dgt_normalized = test_3dgt_normalized.cuda()


        inp_test_poses = test_poses_2dgt_normalized

        pred_test, _ = depth_estimator(inp_test_poses)

        if (config['model'] == 'poseformer'):
            pred_test = pred_test.squeeze()
            inp_test_poses = inp_test_poses[:, 4, :]

        pred_test[:, 0] = 0.0

        pred_test_depth = pred_test + config["depth"]
        pred_test_poses = torch.cat(
            ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
            pred_test_depth), dim=1).detach().cpu().numpy()
        
        results['input_2d'] = inp_test_poses.cpu().detach().numpy()
        results['gt_3d'] = test_3dgt_normalized.cpu().detach().numpy()
        results['pred_3d'] = pred_test_poses

        import pickle
        with open('elepose_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        #print(pred_test_poses.shape)
        #print(inp_test_poses.shape)
        #print(test_3dgt_normalized.shape)
        exit()
        # rotate to camera coordinate system
        test_poses_cam_frame = pred_test_poses.reshape(-1, 3, 17)
        

        losses_pa = mb().pmpjpe(test_3dgt_normalized,
                                        torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17
                                        ).cpu().numpy()
        
        losses_mpjpe_scaled = mb().mpjpe(test_3dgt_normalized,
                                        torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17,
                                        root_joint=0).cpu().numpy()
        
        losses_pck = mb().PCK(test_3dgt_normalized,
                                        torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17,
                                        root_joint=0).cpu().numpy()
        
        losses_auc = mb().AUC(test_3dgt_normalized,
                                        torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17,
                                        root_joint=0).cpu().numpy()
        
        '''
        print("Scaled MPJPE: ", losses_mpjpe_scaled)
        print("PA-MPJPE: ",losses_pa)
        print("PCK: ",losses_pck)
        print("AUC: ",losses_auc)
        '''
        mpjpe_loss_per_epoch.extend(losses_mpjpe_scaled)
        pmpjpe_loss_per_epoch.extend(losses_pa)
        pck_loss_per_epoch.append(losses_pck)
        auc_loss_per_epoch.append(losses_auc)

    avg_mpjpe_loss = np.mean(np.array(mpjpe_loss_per_epoch))
    avg_pmpjpe_loss = np.mean(np.array(pmpjpe_loss_per_epoch))
    avg_pck_loss = np.mean(np.array(pck_loss_per_epoch))
    avg_auc_loss = np.mean(np.array(auc_loss_per_epoch))

    print("=======================================")
    print("AVG Scaled MPJPE: ", avg_mpjpe_loss)
    print("AVG PA-MPJPE: ", avg_pmpjpe_loss)
    print("AVG PCK: ", avg_pck_loss)
    print("AVG AUC: ", avg_auc_loss)
    print("=======================================")