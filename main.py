#%%
import numpy as np
from pr3_utils import *
from scipy.linalg import expm, sinm, cosm
from tqdm import trange
import yaml
#%%

with open("./config/config.yml","r") as stream:
    config = yaml.safe_load(stream)
    
dataset = config["DATASET"]
data_path = config["DATASET_PATH"]
figure_path = config["FIGURES_PATH"]

filename = data_path + str(dataset) +  ".npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

#%% 
tau = np.zeros((t.shape[1]-1,1),dtype=np.float64)
u_t = np.zeros((t.shape[1]-1,6),dtype=np.float64)

for i in range(1,t.shape[1]):
    tau[i-1,:] = t[:,i]-t[:,i-1]
    u_t[i-1,:] = np.concatenate([linear_velocity[:,i],angular_velocity[:,i]])

#angle2twist 
twist_t = axangle2twist(u_t)
u_hat_t = axangle2adtwist(u_t)


twist_tau = np.zeros(twist_t.shape)
u_hat_tau = np.zeros(u_hat_t.shape)


for i in range(twist_t.shape[0]):
    twist_tau[i,:] = twist_t[i,:]*tau[i]
    u_hat_tau[i,:] = u_hat_tau[i,:]*tau[i]

for i in range(twist_t.shape[0]):
    exp_u_hat = expm(-u_hat_tau[i,:])
    

# Covariance Noise
W = 0.00001*np.eye(6)

# twist to pose
tau_u_exp = twist2pose(twist_tau)
mu_t_t = np.eye(4)
cov_t_t = np.eye(6)
traj_pose = np.zeros(tau_u_exp.shape,dtype=np.float64)

# traj_pose = []
for k in range(t.size-1):
    mu_t_1_t = mu_t_t@tau_u_exp[k,:]
    traj_pose[k,:] = mu_t_1_t
    mu_t_t = mu_t_1_t
    
traj = np.transpose(traj_pose,(1,2,0))

fig, ax = visualize_trajectory_2d(traj,path_name="trajectory",show_ori=False)
fig.savefig(figure_path + "Traj.png")


# %%
#Ks 
Ks = np.zeros((4,4))
Ks[0,:3] = K[0,:]
Ks[1,:3] = K[1,:]
Ks[2,:3] = K[0,:]
Ks[2, 3] = -b*K[0,0]
Ks[3,:3] = K[1,:]
Number_of_samples = 20
downsample_idx = [idx for idx in range(0,features.shape[1],Number_of_samples)]
M = len(downsample_idx)
mu_t = np.zeros((3,M))
# print(mu_t.shape)
mu_homog_t = np.zeros((4,M))
mu_homog_t[3,:]=1

cov_slam_t = 0.0001*np.eye((3*M+6))
# print(cov_slam_t.shape)
P = np.zeros((3,4))
for i in range(3):
    P[i][i]=1
cam_T_imu = np.linalg.inv(imu_T_cam)

# %%
old_index_track = set()

for k in trange(t.size-1):
    # m_valid_idx = (features[:,:,k]>0)
    mu_homog_t = np.vstack((mu_t,np.ones((1,M))))
    # print(mu_homog_t[:,1])
    # break
    valid_feature_idx = []
    features_valid = []
    indx_H = []
    downsample_idx = [idx for idx in range(0,features.shape[1],Number_of_samples)]
    features_downsampled = features[:,downsample_idx,k]
    for i in range(features_downsampled.shape[1]):
        if(features_downsampled[:,i][0] >= 0):
            # valid_feature_idx.append(i)
            features_valid.append(features_downsampled[:,i])
            indx_H.append(i)
            
    m_final = np.array(features_valid)
    # print(m_final.shape)
    # break
    if(len(indx_H)==0):
        continue
    idx_new = [indx for indx in range(len(indx_H)) if indx_H[indx] not in old_index_track]
    idx_new_mut = [indx_H[indx] for indx in range(len(indx_H)) if indx_H[indx] not in old_index_track]
    
    idx_old = [indx_H[indx] for indx in range(len(indx_H)) if indx_H[indx] in old_index_track]
    idx_old_mfinal = [indx for indx in range(len(indx_H)) if indx_H[indx] in old_index_track]
    
    for kk in indx_H:
        old_index_track.add(kk)
       
    z_op = Ks[2,3]/(m_final[:,2]-m_final[:,0])
    
    # u_L = Ks[0,0]*x_op + Ks[0,2]*z_op
    # v_L = Ks[1,1]*y_op + Ks[1,2]*z_op
    x_op = (m_final[:,0] - Ks[0,2]*z_op)/Ks[0,0]
    y_op = (m_final[:,1] - Ks[1,2]*z_op)/Ks[1,1]
    

    XYZ_o = np.vstack([x_op,y_op,z_op,np.ones(x_op.shape)])
    
    XYZ_w = traj[:,:,k] @ imu_T_cam @ XYZ_o
    
    if len(idx_new)!=0:
        mu_t[:,idx_new_mut] = XYZ_w[:3,idx_new]

    Nt = len(indx_H)
    
    # noise_covariance = np.eye(4)
    
    
    if len(idx_old)==0:
        # print(k)
        continue
    else:
        #do update step here
        # print(k)
        H_slam_t_1 = np.zeros((4*len(idx_old),3*M+6))
        # Z_t_1_obs = m_final[idx_old,:]
        # print(Z_t_1_obs)
        
        Z_t_1_obs = m_final[idx_old_mfinal,:].flatten()
        # print(Z_t_1_obs)
        # break
        V = np.eye(4*len(idx_old))
        # print(Z_t_1_obs)
        # print(Z_t_1_obs)
        # print("OBSERVATIONS")
        # break
        
        # mu_homog_t[:,idx_old] = XYZ_w[:,idx_old]
        # print(mu_homog_t.shape)
        # break
        inv_T_t_1 = np.linalg.inv(traj[:,:,k])
        mu_homog_t_Nt = mu_homog_t[:,idx_old]
        # print(mu_homog_t_Nt.shape)
        # break
        ph = cam_T_imu@inv_T_t_1@mu_homog_t_Nt
        # 
        Z_t_1 = Ks@projection(ph.T).T
        
        # print(Z_t_1)
        # break
        
        Z_t_1 = (Z_t_1.T).flatten()
        
        # print(Z_t_1)
        
        # break
        # Z_t_1 = Z_t_1[:3,:]
        
        MulFac = cam_T_imu@inv_T_t_1@P.T # 4x3
        
        for j in range(len(idx_old)):
            H_slam_t_1[4*j:4*j+4,3*idx_old[j]:3*idx_old[j]+3] = Ks@projectionJacobian(ph[:,j].T)@MulFac
        # test = np.linalg.inv(H_t_1@cov_slam_t@H_t_1.T + V)
        K_t_1 = cov_slam_t@(H_slam_t_1.T)@(np.linalg.inv(H_slam_t_1@cov_slam_t@H_slam_t_1.T + V))
        mu_t_flatten = (mu_t.T).flatten()
        
        mu_t_flatten = mu_t_flatten + K_t_1@(Z_t_1_obs - Z_t_1)
        # print("DONE")
        # break
        cov_slam_t = (np.eye(3*M) - K_t_1@H_slam_t_1)@cov_slam_t
        # break
    mu_t = np.reshape(mu_t_flatten,(M,3)).T
    # print(H_t_1.shape)
    # break 
   


# %%
def visualize_trajectory_2d_map(pose,mu_t,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    ax.scatter(mu_t[0,:],mu_t[1,:],marker='.',s=2,label="landmarks")
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax


# %%
fig, ax = visualize_trajectory_2d_map(traj,mu_t,path_name="Unknown",show_ori=False)
fig.savefig("VisualMap.png")



