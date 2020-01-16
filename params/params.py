#K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape

#number of frames to load
K = 8

#path to main weight
path_to_chkpt = 'model_weights.tar' 

#path to backup
path_to_backup = 'backup_model_weights.tar'

#CHANGE first part
path_to_Wi = "/mnt/ACA21355A21322FE/VoxCeleb/"+"Wi_weights"
#path_to_Wi = "test/"+"Wi_weights"

#CHANGE if better gpu
batch_size = 2

#dataset save path
path_to_preprocess = '/mnt/ACA21355A21322FE/VoxCeleb/saves3'

#default for Voxceleb
frame_shape = 224
