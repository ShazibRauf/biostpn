import torch
import torch.nn as nn
import numpy as np


def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
    print("Scale:", scale.mean())
    scale = 145.5329587164913
    #p2ds = poses_2d / scale.mean()
    p2ds = poses_2d / scale
    
    p2ds = p2ds * (1 / 10)

    return p2ds


def normalize_head_test(poses_2d, scale=145.5329587164913):  # ground truth
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]

    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds


def perspective_projection(pose_3d):

    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))


def create_sequences(poses_2d):
    # Creating an example batch of arrays (256, 17, 34)
    batch_size = poses_2d.shape[0]

    # Define the parameters
    take = 9  # Number of elements to take
    step = 1  # Step size

    # Calculate the shape of the resulting array
    new_shape = (batch_size, (poses_2d.shape[1] - take) // step + 1, take, poses_2d.shape[2])

    # Initialize the new array with zeros
    result_poses_2d_batch = np.zeros(new_shape)

    # Fill the new batch by taking 9 elements with a step of 1
    for i in range(0, poses_2d.shape[1] - take + 1, step):
        result_poses_2d_batch[:, i // step] = poses_2d[:, i:i + take]

    print("Original Batch shape:", poses_2d.shape)
    print("Resulting Batch shape:", result_poses_2d_batch.shape)

    return result_poses_2d_batch


def bone_symmtery_loss(poses):

    # Define the joint pairs for which you want to calculate bone length symmetry loss
    joint_pairs = [
        ((0, 4), (0, 1)),
        ((1, 2), (4, 5)),
        ((2, 3), (5, 6)),
        ((8, 14), (8, 11)),
        ((14, 15), (11, 12)),
        ((15, 16), (12, 13)),
    ]

    # Extract the joint pairs and reshape for broadcasting
    left_joints = torch.tensor([pair[0] for pair in joint_pairs])  # Shape: (6, 2)
    right_joints = torch.tensor([pair[1] for pair in joint_pairs])  # Shape: (6, 2)

    # Calculate the bone lengths for both sides of each joint pair using broadcasting
    left_bone_lengths = torch.norm(poses[:, left_joints[:, 0]] - poses[:, left_joints[:, 1]], dim=2)  # Shape: (512, 6)
    right_bone_lengths = torch.norm(poses[:, right_joints[:, 0]] - poses[:, right_joints[:, 1]], dim=2)  # Shape: (512, 6)

    # Calculate the squared differences in bone lengths
    losses = (left_bone_lengths - right_bone_lengths) ** 2  # Shape: (512, 6)

    # Calculate the mean squared error loss for bone length symmetry across all poses
    mean_loss = torch.mean(losses)
    return mean_loss.item()


def calculate_angles(points1, points2, points3):
    vectors1 = points1 - points2
    vectors2 = points3 - points2

    dot_products = torch.sum(vectors1 * vectors2, dim=-1)
    magnitudes1 = torch.norm(vectors1, dim=-1)
    magnitudes2 = torch.norm(vectors2, dim=-1)

    cosine_thetas = dot_products / (magnitudes1 * magnitudes2)
    angle_rads = torch.acos(torch.clamp(cosine_thetas, -1.0, 1.0))
    
    # Convert radians to degrees by multiplying by 180/pi
    angle_degrees = angle_rads * (180.0 / 3.141592653589793)

    return angle_degrees


def bone_angle_loss(min_angle, max_angle, y_pred):
    
    # Create a mask to identify angles outside the desired range
    outside_range_mask = (y_pred < min_angle) | (y_pred > max_angle)
    
    # Apply a penalty to angles outside the desired range
    penalty = torch.where(outside_range_mask, torch.tensor(1000.0), torch.tensor(0.0))
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss


# significantly before storing a new example.
def select_frame_indices_to_include(poses_3d):
    # Take only frames where movement has occurred for the protocol #2 train subjects
    frame_indices = []
    prev_joints3d = None
    threshold = 40 ** 2  # Skip frames until at least one joint has moved by 40mm
    for i, joints3d in enumerate(poses_3d):
        if prev_joints3d is not None:
            max_move = ((joints3d - prev_joints3d) ** 2).sum(axis=-1).max()
            if max_move < threshold:
                continue
        prev_joints3d = joints3d
        frame_indices.append(i)
    return np.array(frame_indices)