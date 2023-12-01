from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from sklearn.decomposition import PCA
from utils.helpers import *



class H36MDataset(Dataset):

    def __init__(self, fname='', get_PCA=False, normalize_func=None, stage=None, poses_2d=None, poses_3d=None, pairs=None, elevation=None, receptive_field=0):
        self.receptive_field = receptive_field
        self.pad = (self.receptive_field - 1) // 2 
        self.chunk_length = 1
        self.stage = stage
        
        pickle_off = open(fname, "rb")
        loaddata = pickle.load(pickle_off)

        self.poses_2d = loaddata['poses_2d']
        self.poses_3d = loaddata['poses_3d']
    
        assert self.poses_3d is None or len(self.poses_3d) == len(self.poses_2d), (len(self.poses_3d), len(self.poses_2d))
        # Build lineage info
        pairs = [] # (seq_idx, start_frame) tuples
        for i in range(len(self.poses_2d)):
            assert self.poses_2d is None or self.poses_2d[i].shape[0] == self.poses_2d[i].shape[0]
            if (stage == 'train'):
                # Take only frames where movement has occurred
                bounds = select_frame_indices_to_include(self.poses_3d[i])
            elif (stage == 'test'):
                n_chunks = self.poses_2d[i].shape[0]
                #Take after every 64 frame
                #bounds = np.arange(0, n_chunks+1, 64)
                bounds = np.arange(0, n_chunks)

            pairs += zip(np.repeat(i, len(bounds)), bounds)

        self.pairs = pairs
        print("No of chunks: ", len(self.pairs))

        #Flattening the 2D poses for PCA
        flattened_poses = []
        seq_frames_len = []
        for x in range(len(self.poses_2d)):
            seq_frames_len.append(len(self.poses_2d[x]))
            for y in range(len(self.poses_2d[x])):
                flattened_poses.append(self.poses_2d[x][y])

        flattened_poses = torch.tensor(np.array(flattened_poses), dtype=torch.float)
        print("No of samples:", flattened_poses.shape[0])

        #Normalizing the 2D Poses
        normalized_flattened_poses = normalize_func(flattened_poses.permute(0, 2, 1).reshape(-1, 2*17)).numpy()

        #Converting it back into the orignal shape
        itr = 0
        reshaped_poses_2d = []    
        for x in range(len(self.poses_2d)):
            reshaped_poses_2d.append(normalized_flattened_poses[itr:(itr+seq_frames_len[x]), :])
            itr = itr + seq_frames_len[x]

        self.poses_2d = reshaped_poses_2d

        #Applying PCA
        if (get_PCA):
            #self.pca = PCA()
            #self.pca.fit(normalized_flattened_poses)
            import pickle as pk
            self.pca = pk.load(open("/netscratch/satti/experiments/Elepose-Poseformer-correct-dataset/models/pca.pkl",'rb'))


    def __len__(self):
        return len(self.pairs)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = dict()

        chunks = self.pairs[idx]
        #print(type(chunks))

        batch_2d = np.empty((self.receptive_field, self.poses_2d[0].shape[-1]))
        seq_i, frame_idx = chunks
        start_2d = frame_idx - self.pad 
        end_2d = frame_idx + self.pad + 1

        # 2D poses
        seq_2d = self.poses_2d[seq_i]

        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0)), 'edge')
        else:
            batch_2d = seq_2d[low_2d:high_2d]


        batch_3d = np.empty((self.poses_3d[0].shape[-2], self.poses_3d[0].shape[-1]))
        
        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            batch_3d = seq_3d[frame_idx]
        
        sample['poses_3d'] = torch.tensor(batch_3d.transpose(1, 0).reshape(3*17), dtype=torch.float)
        sample['p2d_gt'] = torch.tensor(batch_2d, dtype=torch.float)
        sample['chunks'] = chunks
        return sample
