import os
import shutil

def prepare_data(data_dir,train_data,eval_data,radius):
    
    data_files = os.listdir(data_dir)
    mrc_files = [os.path.join(data_dir,f)  for f in data_files if f.endswith('.mrc')]
    coords_files = [os.path.join(data_dir,f)  for f in data_files if f.endswith('.txt')]

    # Check if train and eval data exist in the mrc files
    if train_data not in mrc_files:
        raise ValueError(f"Train data file {train_data} not found in .mrc files")
    if eval_data not in mrc_files:
        raise ValueError(f"Eval data file {eval_data} not found in .mrc files")

    # Define directory paths
    train_dir = os.path.join(data_dir, 'train')
    eval_dir = os.path.join(data_dir, 'eval')
    test_dir = os.path.join(data_dir, 'test')

    # Remove existing directories if they exist
    for dir_path in [train_dir, eval_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed existing directory: {dir_path}")
            os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
    
    shutil.copy(train_data,train_dir)
    
    for mrc_file in mrc_files:
            shutil.copy(
                mrc_file,
                test_dir)

    train_name = train_data.split('/')[-1].split('.')[0]
    eval_name = eval_data.split('/')[-1].split('.')[0]
    train_coords = data_dir + '/' + train_name + '.txt'
    eval_coords = data_dir + '/' + eval_name + '.txt'
    
    if train_coords not in coords_files:
        raise ValueError(f"Train data file {train_coords} not found in .mrc files")
    if eval_coords not in coords_files:
        raise ValueError(f"Eval data file {eval_coords} not found in .mrc files")

    shutil.copy(train_coords,train_dir)
    shutil.copy(eval_coords,eval_dir)

    tomo_occupancy_dir = os.path.join(eval_dir,'tomo_occupancy')
    os.makedirs(tomo_occupancy_dir)
    shutil.copy(eval_coords,eval_dir)
    shutil.copy(eval_coords,tomo_occupancy_dir)

    import pandas as pd
    import mrcfile
    eval_coords_data = pd.read_csv(eval_coords,sep = '\t',header=None,
                                   names=['x', 'y', 'z'])
    max_z = eval_coords_data['z'].max() + radius + 2
    
    eval_mrc_path = data_dir + '/' + 'eval' + '/' + eval_name + '.mrc'
    with mrcfile.open(eval_data,permissive=True) as f:
        tomo_data = f.data[:max_z,:,:]

        with mrcfile.new(eval_mrc_path,overwrite=True) as mrc_out:
            mrc_out.set_data(tomo_data)
            # Update header to match new dimensions
            mrc_out.header.nz =  tomo_data.shape[0] # Update z dimension
            mrc_out.header.cella = f.header.cella  # Keep original cell dimensions
            mrc_out.header.mode = f.header.mode  # Preserve data mode
            
            # Optional: Update other relevant header fields
            mrc_out.update_header_stats()
    
    from build_target import TargetBuilder
    target_builder= TargetBuilder()
    occupancy_map = target_builder.generate_with_spheres(eval_coords_data,tomo_data,radius)
    occupancy_map_path = tomo_occupancy_dir + '/' + eval_name + '.mrc'
    with mrcfile.new(occupancy_map_path,overwrite=True) as f:
        f.set_data(occupancy_map)

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # project options
    
    parser.add_argument('--tomo_dir', type=str, default='./deconv', help='the directory for tomograms ')
    parser.add_argument("--train_data", type=str, default=None, help="the path of the tomograms for training")
    parser.add_argument("--eval_data", type=str, default=None, help="the path of the tomograms for validation")
    parser.add_argument("--radius", type=int, default=None, help="the radius in voxel of protein particle in tomograms")
    
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = create_parser()

    prepare_data(data_dir=opt.tomo_dir,
                 train_data=opt.train_data,
                 eval_data=opt.eval_data,
                 radius=opt.radius)
    

