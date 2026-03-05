

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os 
import numpy as np 
from scipy.io import loadmat
from tqdm import tqdm 
import time 
import pickle 
from pathlib import Path 

from src import load_mesh, EITFEM, RegGaussNewton, interpolateRecoToPixGrid, image_to_mesh
from src import create_phantoms, LinearisedRecoFenics, HanddrawnImages

level = 6
num_images = 2000
test = False 
use_handdrawn_images = True 


level_to_alphas = {
    1 : [[1956315.789, 0.,0.],[0., 656.842 , 0.],[0.,0.1,6.105],[1956315.789/3., 656.842/3,6.105/3.], [1e4, 0.1,5.]], # just test values, to be decided upon [tv, sm, lm]
    2 : [[1890000, 0.,0.],[0., 505.263, 0.],[0.,0.1,12.4210],[1890000/3., 505.263/3.,12.421/3.], [1e4, 0.1,5.]], 
    3 : [[1890000, 0.,0.],[0., 426.842, 0.],[0.,0.1,22.8421],[2143157/3., 426.842/3.,22.8421/3.], [6e5, 3,14]],
    4 : [[1890000, 0.,0.],[0., 1000., 0.],[0.,0.1,43.052],[1890000/3., 1000./3.,43.052/3.], [6e5, 8,16]], 
    5 : [[1890000, 0.,0.],[0., 843.6842, 0.],[0.,0.1,30.7368],[1890000/3., 843.684/3.,30.7368/3.], [6e5, 10,18]], 
    6 : [[40000, 0.,0.],[0., 895.789, 0.],[0.,0.1,74.947],[40000/3., 895.78/3.,74.947/3.], [6e5, 25,20]], 
    7 : [[40000, 0.,0.],[0., 682.105, 0.],[0.,0.1,18.421],[40000/3., 687.3684/3.,18.421/3.], [6e5, 30,22]], 
}

alphas = level_to_alphas[level]


#base_path = "/localdata/AlexanderDenker/KTC2023/dataset/level_" + str(level)
if test:
    base_path = "/localdata/AlexanderDenker/KTC2023/dataset/level_" + str(level)  #"/pvfs2/adenker/KTC2023/dataset_test/level_" + str(level)
else:
    base_path = "/localdata/AlexanderDenker/KTC2023/dataset/level_" + str(level)  #"/pvfs2/adenker/KTC2023/dataset/level_" + str(level) #"/localdata/AlexanderDenker/tmp" #"/pvfs2/adenker/KTC2023/dataset/level_" + str(level)

gt_path = Path(os.path.join(base_path, "gt"))
measurement_path = Path(os.path.join(base_path, "measurements"))
reco_path = Path(os.path.join(base_path, "gm_reco"))

gt_path.mkdir(parents=True, exist_ok=True)
measurement_path.mkdir(parents=True, exist_ok=True)
reco_path.mkdir(parents=True, exist_ok=True)


def check_highest_digit(base_path):
    file_list_int = [int(f.split(".")[0].split("_")[-1]) for f in os.listdir(base_path)]
    if len(file_list_int) == 0:
        return 0
    else:
        return max(file_list_int)

max_image_idx = 2000 #check_highest_digit(reco_path) + 1

print("max_image_idx: ", max_image_idx)

y_ref = loadmat('TrainingData/ref.mat') #load the reference data
Injref = y_ref["Injref"]
Mpat = y_ref["Mpat"]

mesh, mesh2 = load_mesh("Mesh_dense.mat")

Nel = 32
z = (1e-6) * np.ones((Nel, 1))  # contact impedances
vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include 

### simulate measurements with KTC solver
solver = EITFEM(mesh2, Injref, Mpat, vincl)

noise_std1 = 0.05  # standard deviation of the noise as percentage of each voltage measurement
noise_std2 = 0.01  # %standard deviation of 2nd noise component (this is proportional to the largest measured value)
solver.SetInvGamma(noise_std1, noise_std2, y_ref["Uelref"])  # compute the noise precision matrix

# simulate measurements of empty watertank
sigma_background = np.ones((mesh.g.shape[0], 1))*0.745
Uelref = solver.SolveForward(sigma_background, z)
noise = solver.InvLn * np.random.randn(Uelref.shape[0],1)
Uelref = Uelref + noise

### solve using Jacobian from Fenics solver 
mesh_name = "sparse"
B = Mpat.T

vincl_level = np.ones(((Nel - 1),76), dtype=bool) 
rmind = np.arange(0,2 * (level - 1),1) #electrodes whose data is removed

#remove measurements according to the difficulty level
for ii in range(0,75):
    for jj in rmind:
        if Injref[jj,ii]:
            vincl_level[:,ii] = 0
        vincl_level[jj,:] = 0

reconstructor = LinearisedRecoFenics(Uelref, B, vincl_level, mesh_name=mesh_name)

if use_handdrawn_images:
    dataset = HanddrawnImages(path_to_images = "/home/adenker/projects/ktc2023/dl_for_ktc2023/data/KTC_handdrawn_images", rotate=True)


for i in tqdm(range(num_images)):
    full_time_1 = time.time() 
    
    if use_handdrawn_images:
        img_idx = np.random.randint(len(dataset))
        sigma_pix = dataset[img_idx]
    else:
        sigma_pix = create_phantoms()

    img_name = "gt_ztm_{:06d}.npy".format(max_image_idx + i)     # gt_000001.npy 
    u_name = "u_ztm_{:06d}.npy".format(max_image_idx + i)
    sigmavalues_name = "sigmavalues_ztm_{:06d}.pkl".format(max_image_idx + i)
    reco_name = "recos_ztm_{:06d}.npy".format(max_image_idx + i)
    
    np.save(os.path.join(gt_path, img_name), sigma_pix)


    # background conductivity  0.745
    background = 0.745

    # resistive between 0.025 - 0.125
    resistive = np.random.rand()*0.1 + 0.025

    # conductive between 5.0 and 6.0
    conductive = np.random.rand() + 5.0

    sigma = np.zeros(sigma_pix.shape)
    sigma[sigma_pix == 0.0] = background
    sigma[sigma_pix == 1.0] = resistive
    sigma[sigma_pix == 2.0] = conductive

    sigma_gt = image_to_mesh(np.flipud(sigma).T, mesh)

    time1 = time.time()
    Uel_sim = solver.SolveForward(sigma_gt, z)
    noise = solver.InvLn * np.random.randn(Uel_sim.shape[0],1)
    Uel_noisy = Uel_sim + noise
    time2 = time.time() 

    print("Simulate Measurements: ", time2-time1, "s")

    measurement_dict = {
        'background': background,
        'resistive': resistive,
        'conductive': conductive
    }

    with open(os.path.join(measurement_path, sigmavalues_name),'wb') as f:
        pickle.dump(measurement_dict, f)

    np.save(os.path.join(measurement_path, u_name), Uel_noisy)

    
    time_1 = time.time() 
    delta_sigma_list = reconstructor.reconstruct_list(Uel_noisy, alphas)
    time_2 = time.time()         
    print("Reconstruction: ", time_2 - time_1, "s")


    delta_sigma_0 = reconstructor.interpolate_to_image(delta_sigma_list[0])
    delta_sigma_1 = reconstructor.interpolate_to_image(delta_sigma_list[1])
    delta_sigma_2 = reconstructor.interpolate_to_image(delta_sigma_list[2])
    delta_sigma_3 = reconstructor.interpolate_to_image(delta_sigma_list[3])
    delta_sigma_4 = reconstructor.interpolate_to_image(delta_sigma_list[4])

    """
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5)

    im = ax0.imshow(sigma)
    fig.colorbar(im, ax=ax0)
    ax0.axis("off")
    ax0.set_title("GT")

    im = ax1.imshow(delta_sigma_0, cmap="jet")
    fig.colorbar(im, ax=ax1)
    ax1.axis("off")
    ax1.set_title("TV-L2")

    im = ax2.imshow(delta_sigma_1, cmap="jet")
    fig.colorbar(im, ax=ax2)
    ax2.set_title("Smoothness Prior")
    ax2.axis("off")

    im = ax3.imshow(delta_sigma_2, cmap="jet")
    fig.colorbar(im, ax=ax3)
    ax3.axis("off")
    ax3.set_title("Levenberg-Marquardt")

    im = ax4.imshow(delta_sigma_3, cmap="jet")
    fig.colorbar(im, ax=ax4)
    ax4.axis("off")
    ax4.set_title("Combined Prior")

    plt.show()
    """
    sigma_reco = np.stack([delta_sigma_0, delta_sigma_1, delta_sigma_2, delta_sigma_3, delta_sigma_4])
    print("SIGMA RECO: ", sigma_reco.shape)

    np.save(os.path.join(reco_path, reco_name), sigma_reco)


    print("Full time for one training sample: ", time.time() - full_time_1, "s")