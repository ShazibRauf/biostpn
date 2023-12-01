import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_h36m_fetch_all import H36MDataset
from models_def import DepthAngleEstimator, DepthAngleEstimator_transformer
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics
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

#torch.manual_seed(1234)
#np.random.seed(1234)

print("numpy seeding value: ",np.random.get_state()[1][0])
print("torch seeding value: ",torch.initial_seed())

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
    "weight_bl": 50,
    "weight_2d": 1.0,
    "weight_3d": 1.0,
    "weight_velocity": 1.0,
    "weight_bl_symmetry": 50,
    "weight_bone_angle": 1,
    "weight_NF": 1.0,
    "depth": args.translation,
    "use_gt": True,
    "num_joints": 17,
    "num_bases": num_bases,
    "trainfile": data_folder + 'H36M/h36m_train_detections.pkl',
    "testfile": data_folder + 'H36M/h36m_test_detections.pkl',
    "checkpoints": project_folder + 'checkpoints/',
    "receptive_field": 17,
    "model": "poseformer",
    "weight_decay": 1e-5,
    "drop_out": 0.0
}

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
                                    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=config['drop_out'])


if torch.cuda.is_available():
    print("CUDA Available")
    depth_estimator = nn.DataParallel(depth_estimator)
    depth_estimator = depth_estimator.cuda()
    inn_2d_1 = inn_2d_1.cuda()


# Define your data loaders and dataset classes.
train_dataset = H36MDataset(config["trainfile"], get_PCA=True, normalize_func=normalize_head, stage="train", receptive_field=config['receptive_field'])

if config["use_gt"]:
    test_dataset = H36MDataset(config["testfile"], get_PCA=False, normalize_func=normalize_head_test, stage="test", receptive_field=9)
else:
    test_dataset = H36MDataset(config["testfile"], get_PCA=False, normalize_func=normalize_head_test, stage="test", receptive_field=9)

pca = train_dataset.pca

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, num_workers=16)

bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                    0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()


optimizer = torch.optim.Adam(depth_estimator.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

losses_3d_train = []  
ls_likeli = []
ls_L3d = []
ls_rep_rot = []
ls_re_rot_3d = []
ls_bl_prior = []
mpjpe_loss = []


epoch = 0
while epoch < config["N_epochs"]:

    
    N = 0
    epoch_loss = 0
    epoch_likeli = 0
    epoch_L3d = 0
    epoch_rep_rot = 0
    epoch_re_rot_3d = 0
    epoch_bl_prior = 0
    epoch_bl_symmetry = 0

    #Training Mode
    depth_estimator.train()
    # Iterate through the DataLoader
    for train_batch in tqdm(train_loader, desc=f"Stage1 - Epoch {epoch} "):
        
        inputs_2d = train_batch['p2d_gt'] #(256, 17, 34)

        optimizer.zero_grad()

        pred_3d_seq = torch.zeros([9, inputs_2d.shape[0], 51])
        props_3d_seq = torch.zeros([9, inputs_2d.shape[0], 1])
        for x in range(9):
            input_2d_seq = inputs_2d[:, x:x+9, :] #(256, 9, 34)
            
            if (config['model'] == 'elepose'):
                input_elepose = input_2d_seq[:, 4, :] #(256, 34)

                if torch.cuda.is_available():
                    input_elepose = input_elepose.cuda()

                pred, props = depth_estimator(input_elepose)
                pred[:, 0] = 0.0

                depth = pred + config['depth']
                depth[depth < 1.0] = 1.0
                pred_3d = torch.cat(((input_elepose.reshape(-1, 2, 17) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1)

                pred_3d_seq[x] = pred_3d
                props_3d_seq[x] = props

            elif(config['model'] == 'poseformer'):
                input_poseformer = input_2d_seq #(256, 9, 34)

                if torch.cuda.is_available():
                    input_poseformer = input_poseformer.cuda()

                pred, props = depth_estimator(input_poseformer)
                #pred = pred.squeeze()
                pred[:, 0] = 0.0

                depth = pred + config['depth']
                depth[depth < 1.0] = 1.0
                pred_3d = torch.cat(((input_poseformer[:, 4, :].reshape(-1, 2, 17) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1)

                pred_3d_seq[x] = pred_3d
                props_3d_seq[x] = props

        
        pred_3d = pred_3d_seq.flatten(0, 1).cuda() #(256*9, 51)
        #props = props_3d_seq.flatten(0, 1).cuda() #(256*9, 1)
        props = props_3d_seq[4, :, :].cuda() #(256, 1)
        
        #props = props[::receptive_field]
        x_ang_comp = torch.ones((props.shape[0], 1)).cuda() * props
        y_ang_comp = torch.zeros((props.shape[0], 1)).cuda()
        z_ang_comp = torch.zeros((props.shape[0], 1)).cuda()

        euler_angles_comp = torch.cat((x_ang_comp, y_ang_comp, z_ang_comp), dim=1)
        R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')

        # sample from learned distribution
        elevation = torch.cat((props_3d_seq.flatten(0, 1).mean().reshape(1), props_3d_seq.flatten(0, 1).std().reshape(1)))
        x_ang = (-elevation[0]) + elevation[1] * torch.normal(torch.zeros((props.shape[0], 1)).cuda(), torch.ones((props.shape[0], 1)).cuda())
        y_ang = (torch.rand((props.shape[0], 1)).cuda() - 0.5) * 2.0 * np.pi
        z_ang = torch.zeros((props.shape[0], 1)).cuda()

        Rx = euler_angles_to_matrix(torch.cat((x_ang, z_ang, z_ang), dim=1), 'XYZ')
        Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')
        R = Rx @ (Ry @ R_comp) #(256, 3, 3)
        R = R.repeat_interleave(9, dim=0) #(256*9, 3, 3)


        #Root Relative
        pred_3d = pred_3d.reshape(-1, 3, 17) - pred_3d.reshape(-1, 3, 17)[:, :, [0]]
        rot_poses = (R.matmul(pred_3d)).reshape(-1, 51)

        ## lift from augmented camera and normalize
        global_pose = torch.cat((rot_poses[:, 0:34], rot_poses[:, 34:51] + config['depth']), dim=1)
        rot_2d = perspective_projection(global_pose)
        norm_poses = rot_2d

        norm_poses_mean = norm_poses[:, 0:34] - torch.Tensor(pca.mean_.reshape(1, 34)).cuda()
        latent = norm_poses_mean @ torch.Tensor(pca.components_.T).cuda()          
        
        z, log_jac_det = inn_2d_1(latent[:, 0:num_bases])
        likelis = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

        losses_likeli = likelis.mean()
        epoch_likeli += latent.shape[0] * losses_likeli.cpu().detach() #<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if (config['model'] == 'elepose'):
            norm_poses = norm_poses.reshape(inputs_2d.shape[0], 9, 34)[:, 4, :]
        elif (config['model'] == 'poseformer'):
            norm_poses = norm_poses.reshape(inputs_2d.shape[0], 9, 34)

        ## reprojection error
        pred_rot, _ = depth_estimator(norm_poses)

        #if (config['model'] == 'poseformer'):
        #    pred_rot = pred_rot.squeeze()

        pred_rot[:, 0] = 0.0

        pred_rot_depth = pred_rot + config['depth']
        pred_rot_depth[pred_rot_depth < 1.0] = 1.0
        if (config['model'] == 'elepose'):
            pred_3d_rot = torch.cat(((norm_poses[:, 0:34].reshape(-1, 2, 17) * pred_rot_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth), dim=1)
        elif (config['model'] == 'poseformer'):
            pred_3d_rot = torch.cat(((norm_poses[:, 4, 0:34].reshape(-1, 2, 17) * pred_rot_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth), dim=1)


        pred_3d_rot = pred_3d_rot.reshape(-1, 3, 17) - pred_3d_rot.reshape(-1, 3, 17)[:, :, [0]]
        losses_L3d = (rot_poses.reshape(inputs_2d.shape[0], 9, 51)[:, 4, :] - pred_3d_rot.reshape(-1, 51)).norm(dim=1).mean()
        epoch_L3d += inputs_2d.shape[0] * losses_L3d.cpu().detach().numpy()
        #print('losses_L3d',losses_L3d)

        R = R.reshape(inputs_2d.shape[0], 9, 3, 3)[:, 4, :, :]
        re_rot_3d = (R.permute(0, 2, 1) @ pred_3d_rot).reshape(-1, 51)
        pred_rot_global_pose = torch.cat((re_rot_3d[:, 0:34], re_rot_3d[:, 34:51] + config['depth']), dim=1)
        re_rot_2d = perspective_projection(pred_rot_global_pose)
        norm_re_rot_2d = re_rot_2d

        
        losses_rep_rot = (norm_re_rot_2d - inputs_2d[:, 8, :].cuda()).abs().sum(dim=1).mean()
        epoch_rep_rot += inputs_2d.shape[0] * losses_rep_rot.cpu().detach().numpy()
        #print('losses_rep_rot', losses_rep_rot)

        pred_3d_def = pred_3d.reshape(inputs_2d.shape[0], 9, 3, 17)[:, 4, :, :] 
        # pairwise deformation loss
        num_pairs = int(np.floor(pred_3d_def.shape[0] / 2))
        pose_pairs = pred_3d_def[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_3d = re_rot_3d[0:(2*num_pairs)].reshape(-1, 2, 51)
        losses_re_rot_3d = ((pose_pairs[:, 0] - pose_pairs[:, 1]) - (pose_pairs_re_rot_3d[:, 0] - pose_pairs_re_rot_3d[:, 1])).norm(dim=1).mean()
        epoch_re_rot_3d += inputs_2d.shape[0] * losses_re_rot_3d.cpu().detach().numpy()
        #print('losses_re_rot_3d', losses_re_rot_3d)

        ## bone symmetry
        losses_bone_symmetry = bone_symmtery_loss(pred_3d.permute(0, 2, 1))
        epoch_bl_symmetry += inputs_2d.shape[0] * losses_bone_symmetry
        '''

        joints = [[4, 5, 6], #'LeftHip’, ’LeftKnee', 'LeftAnkle'
                [1, 2, 3], #’RightHip’, ’RightKnee', ‘RightAnkle'
                [14, 15, 16], #'LeftShoulder', 'LeftElbow', 'LeftWrist'
                [11, 12, 13] #’RightShoulder', ‘RightElbow', ‘RightWrist' 
                ]
        angle_range = [[0.0, 150.0],
                        [0.0, 150.0],
                        [0.0, 160.0],
                        [0.0, 160.0]]
    
        pred_3d_permute = pred_3d.permute(0, 2, 1)
        losses_bone_angle = []
        
        x=0
        for joint in joints:

            # Example: Calculate the angle between joints 1, 2, and 3
            joint1 = pred_3d_permute[:, joint[0], :]
            joint2 = pred_3d_permute[:, joint[1], :]
            joint3 = pred_3d_permute[:, joint[2], :]

            angle_deg = calculate_angles(joint1, joint2, joint3)
            losses_bone_angle.append(bone_angle_loss(angle_range[x][0], angle_range[x][1], angle_deg))
            x=x+1
        '''

        ## bone lengths prior
        bl = get_bone_lengths_all(pred_3d.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        losses_bl_prior = (bone_relations_mean - rel_bl).square().sum(dim=1).mean()
        epoch_bl_prior += inputs_2d.shape[0] * losses_bl_prior.cpu().detach().numpy()
        #print('losses_bl_prior',losses_bl_prior)

        losses_loss = config['weight_NF'] * losses_likeli + \
                    config['weight_2d']*losses_rep_rot + \
                    config['weight_3d'] * losses_L3d + \
                    config['weight_velocity']*losses_re_rot_3d + \
                    config['weight_bl_symmetry']*losses_bone_symmetry

        losses_loss = losses_loss + config['weight_bl']*losses_bl_prior #+ config['weight_bone_angle'] * (losses_bone_angle[0] + losses_bone_angle[1] + losses_bone_angle[2] + losses_bone_angle[3])

        epoch_loss += inputs_2d.shape[0] * losses_loss.item()
        N += inputs_2d.shape[0]

        losses_loss.backward()
        optimizer.step()
        
        del inputs_2d, pred_3d, props
        torch.cuda.empty_cache()
    
    losses_3d_train.append(epoch_loss / N)
    ls_L3d.append(epoch_L3d / N)
    ls_likeli.append(epoch_likeli / (N * 9))
    ls_rep_rot.append(epoch_rep_rot / N)
    ls_re_rot_3d.append(epoch_re_rot_3d / N)
    ls_bl_prior.append(epoch_bl_prior / N)
    

    print('[%d] training_loss %f ls_likeli %f ls_L3d %f ls_rep_rot %f ls_re_rot_3d %f ls_bl_prior %f' % (
                epoch,
                losses_3d_train[-1],
                ls_likeli[-1],
                ls_L3d[-1],
                ls_rep_rot[-1],
                ls_re_rot_3d[-1],
                ls_bl_prior[-1]))

    
    with torch.no_grad():
        #Evaluation mode
        depth_estimator.eval()

        mpjpe_loss_one_epoch = []
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
                #pred_test = pred_test.squeeze()
                inp_test_poses = inp_test_poses[:, 4, :]

            pred_test[:, 0] = 0.0

            pred_test_depth = pred_test + config["depth"]
            pred_test_poses = torch.cat(
                ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
                pred_test_depth), dim=1).detach().cpu().numpy()

            # rotate to camera coordinate system
            test_poses_cam_frame = pred_test_poses.reshape(-1, 3, 17)
            

            losses_mpjpe_scaled = mb().mpjpe(test_3dgt_normalized,
                                            torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device), num_joints=17,
                                            root_joint=0).mean().cpu().numpy()
            print("Scaled MPJPE: ", losses_mpjpe_scaled)
            mpjpe_loss_one_epoch.append(losses_mpjpe_scaled)

        avg_mpjpe_loss = np.mean(np.array(mpjpe_loss_one_epoch))
        print("AVG Scaled MPJPE: ", avg_mpjpe_loss)
        mpjpe_loss.append(avg_mpjpe_loss)

    chk_path = os.path.join(config['checkpoints'], 'epoch_{}.bin'.format(epoch))
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch,
        'lr': scheduler.get_last_lr(),
        'optimizer': optimizer.state_dict(),
        'model_pos': depth_estimator.state_dict(),
    }, chk_path)

    epoch += 1
    scheduler.step()


    plt.figure()
    epoch_x = np.arange(0, len(losses_3d_train)) + 1
    plt.plot(epoch_x, losses_3d_train[0:], color='C0')
    plt.legend(['Training Loss'])
    plt.ylabel('Combine Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'Training_loss.png'))
    plt.close('all')

    plt.figure()
    epoch_x = np.arange(0, len(ls_L3d)) + 1
    plt.plot(epoch_x, ls_L3d[0:], color='C1')
    plt.legend(['Training Loss L3d'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'loss_L3d.png'))
    plt.close('all')

    plt.figure()
    epoch_x = np.arange(0, len(ls_rep_rot)) + 1
    plt.plot(epoch_x, ls_rep_rot[0:], color='C1')
    plt.legend(['Training Loss rep_rot'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'loss_rep_rot.png'))
    plt.close('all')


    plt.figure()
    epoch_x = np.arange(0, len(ls_re_rot_3d)) + 1
    plt.plot(epoch_x, ls_re_rot_3d[0:], color='C1')
    plt.legend(['Training Loss re_rot_3d'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'loss_re_rot_3d.png'))
    plt.close('all')

    plt.figure()
    epoch_x = np.arange(0, len(ls_bl_prior)) + 1
    plt.plot(epoch_x, ls_bl_prior[0:], color='C1')
    plt.legend(['Training Loss bl_prior'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'loss_bl_prior.png'))
    plt.close('all')

    plt.figure()
    epoch_x = np.arange(0, len(ls_likeli)) + 1
    plt.plot(epoch_x, ls_likeli[0:], color='C1')
    plt.legend(['Training Loss ls_likeli'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'loss_ls_likeli.png'))
    plt.close('all')


    plt.figure()
    epoch_x = np.arange(0, len(mpjpe_loss)) + 1
    plt.plot(epoch_x, mpjpe_loss[0:], color='C1')
    plt.legend(['Scaled MPJPE Error'])
    plt.ylabel('Scaled MPJPE')
    plt.xlabel('Epoch')
    plt.xlim((0, epoch))
    plt.savefig(os.path.join(config['checkpoints'], 'mpjpe_loss.png'))
    plt.close('all')