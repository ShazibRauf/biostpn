import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def save_2d_img(pose_2d, name):
    x_coord = pose_2d[:, 0]
    y_coord = pose_2d[:, 1]
    labels = np.arange(0,17)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the keypoints as scatter points with reversed y-coordinates
    ax.scatter(x_coord, -y_coord, marker='o', c='b', label='Body Pose')

    for i in range(len(x_coord)):
        ax.annotate(str(labels[i]), (x_coord[i], -y_coord[i]), textcoords="offset points", xytext=(0,10), ha='center')



    
    connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

    for connection in connections:
        x_values = [pose_2d[connection[0]][0], pose_2d[connection[1]][0]]
        y_values = [-pose_2d[connection[0]][1], -pose_2d[connection[1]][1]]  # Reverse y-coordinates
        ax.plot(x_values, y_values, linestyle='-', linewidth=2, alpha=0.7)

    
    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Body Pose")
    ax.legend()

    # Display the plot
    plt.savefig("./"+name)



def visualize_hip_lock(ground_truth, predicted, file_name):
    # Create a subplot with 2 columns: one for Ground Truth and one for Predicted
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{'type': 'scatter3d'}]])
    

    pred_x = predicted[:, 0] 
    pred_y = predicted[:, 1]
    pred_z = predicted[:, 2]
    
    # Plot ground truth annotations
    fig.add_trace(
        go.Scatter3d(
            x=ground_truth[:, 0], 
            y=ground_truth[:, 1], 
            z=ground_truth[:, 2], 
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Before Rotation',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred_x, 
            y=pred_y, 
            z=pred_z, 
            mode='markers',
            marker=dict(color='red', size=5),
            name='After Rotation',
            showlegend=True
        ),
     row=1, col=1
    )
    
    gt_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
    
    for idx, connection in enumerate(gt_connections):
        fig.add_trace(
            go.Scatter3d(
                x=[ground_truth[connection[0], 0], ground_truth[connection[1], 0]],
                y=[ground_truth[connection[0], 1], ground_truth[connection[1], 1]],
                z=[ground_truth[connection[0], 2], ground_truth[connection[1], 2]],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=[pred_x[connection[0]], pred_x[connection[1]]],
                y=[pred_y[connection[0]], pred_y[connection[1]]],
                z=[pred_z[connection[0]], pred_z[connection[1]]],
                mode='lines',
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=1
        )

    max_range = None

    if not max_range:
        # Correct aspect ratio (https://stackoverflow.com/a/21765085).
        max_range = (
            np.array(
                [
                    pred_x.max() - pred_x.min(),
                    pred_y.max() - pred_y.min(),
                    pred_z.max() - pred_z.min(),
                ]
            ).max()
            / 2.0
        )
    else:
        max_range /= 2
    mid_x = (pred_x.max() + pred_x.min()) * 0.5
    mid_y = (pred_y.max() + pred_y.min()) * 0.5
    mid_z = (pred_z.max() + pred_z.min()) * 0.5

    

    # Update layout for better view
    fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z', 
            # Set xlim, ylim, and zlim here
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range])
        ),
        title='Rotation Visualization'
    )
    
    # Save as an HTML file
    html_filename_separate = "./"+file_name
    fig.write_html(html_filename_separate)




def visualize_body_pose(ground_truth, file_name):
    # Create a subplot with 2 columns: one for Ground Truth and one for Predicted
    print(ground_truth.shape)
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{'type': 'scatter3d'}]])
    
    # Plot ground truth annotations
    fig.add_trace(
        go.Scatter3d(
            x=ground_truth[:, 0], 
            y=ground_truth[:, 1], 
            z=ground_truth[:, 2], 
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Ground Truth',
            showlegend=False
        ),
        row=1, col=1
    )
    
    gt_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
    colors = ['red', 'red', 'red', 'lime', 'lime', 'lime', 'cyan', 'cyan', 'cyan', 'cyan', 'lime', 'lime', 'lime', 'red', 'red', 'red']

    for idx, connection in enumerate(gt_connections):
        fig.add_trace(
            go.Scatter3d(
                x=[ground_truth[connection[0], 0], ground_truth[connection[1], 0]],
                y=[ground_truth[connection[0], 1], ground_truth[connection[1], 1]],
                z=[ground_truth[connection[0], 2], ground_truth[connection[1], 2]],
                mode='lines',
                line=dict(color=colors[idx]),
                showlegend=False
            ),
            row=1, col=1
        )
    # Update layout for better view
    fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Ground Truth and Predicted Annotations'
    )
    
    # Save as an HTML file
    html_filename_separate = "./"+file_name
    fig.write_html(html_filename_separate)

gt = np.array([  -68.3494,    63.2364,    48.3664,   154.9633,  -199.9332,  -151.5210,
          -16.6579,   -10.3834,    21.9922,   -27.1031,     8.0059,   152.9949,
          415.1838,   347.5352,  -117.3144,  -395.5033,  -631.7601,  -393.0260,
         -406.0236,    48.1723,   494.2513,  -380.0287,    88.9284,   530.2685,
         -622.6933,  -867.5917,  -955.4758, -1037.8507,  -818.6322,  -837.3045,
         -964.3148,  -832.1476,  -868.2766,  -968.3410,  4956.7012,  5010.1328,
         5184.0190,  5132.7646,  4903.2705,  5023.6030,  5020.8643,  4844.2285,
         4763.1025,  4827.3066,  4755.1489,  4815.5806,  4962.2573,  5176.2944,
         4722.4800,  4613.3394,  4587.0767])

visualize_body_pose(gt.reshape(3, 17).transpose(1, 0), "plot1.html")

gt1 = np.array([ 0.0000,  0.0212,  0.0186,  0.0353, -0.0217, -0.0132,  0.0084,  0.0094,
         0.0147,  0.0066,  0.0124,  0.0365,  0.0779,  0.0647, -0.0088, -0.0574,
        -0.0985,  0.0000, -0.0014,  0.0711,  0.1408,  0.0014,  0.0779,  0.1483,
        -0.0393, -0.0818, -0.0943, -0.1102, -0.0722, -0.0710, -0.0850, -0.0771,
        -0.0864, -0.1041])

save_2d_img(gt1.reshape(2, 17).transpose(1, 0), "testing .png")