import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--subject",
        type=str,
        default=None)
    
    parser.add_argument(
        '--img_count',
        type=int,
        default=0,
        help= 'First N images will be used in the experiment if N is provided. Otherwise all images will be used.'
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    subject = args.subject
    atlasname = 'streams'
    
    print(subject)

    savedir = f'../nsd-results/mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    # Sessions, counting from 1 to 38.
    selected_sessions = range(1, 38)
    
    nsda = NSDAccess('../nsd/')
    nsd_expdesign = scipy.io.loadmat('../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Most nsd_expdesign indices are 1-base.
    shared_img_indexes = nsd_expdesign['sharedix'] -1 
    train_test_split_indexes = shared_img_indexes

    behaviours = pd.DataFrame()
    for i in selected_sessions:
        behaviour = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behaviours = pd.concat((behaviours,behaviour))

    # 1-base index again.
    image_stims_all = behaviours['73KID'] - 1
    image_stims_unique = image_stims_all.unique()

    save_img_stims(savedir, subject, image_stims_all, image_stims_unique, args.img_count, train_test_split_indexes)

    for session in selected_sessions:
        print(f'SESSION: {session}')
              
        beta_trial = nsda.read_betas(subject=subject, 
                                session_index=session, 
                                trial_index=[], # Empty list as index means get all for this session.
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
        if session == 1:
            betas_all = beta_trial
        else:
            betas_all = np.concatenate((betas_all,beta_trial), 0)
    
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    print(betas_all.shape)

    for roi,val in atlas[1].items():
        print(f'ROI:{roi}, VAL: {val}')
        
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi = betas_all[:,atlas[0].transpose([2,1,0])==val]

        print(betas_roi.shape)
        
        # Averaging for each stimulus.
        betas_roi_ave = []
        
        
        # Čia yra vidurkinamos reikšmės. Tarkim pacientas matė tokias nuotraukas: 4, 2, 1, 1, 3, 4, 8, 8.
        # Tai fMRI duomenys, gauti stebint 8-tą nuotrauką, būtų vidurkinami. Analogiškai kiekvienai unikaliai nuotraukai.
        for img_stim_id in image_stims_unique:
            stim_mean = np.mean(betas_roi[image_stims_all == img_stim_id,:],axis=0)
            betas_roi_ave.append(stim_mean)
            
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)
        
        # Train/Test Split.
        save_betas(image_stims_all, train_test_split_indexes, betas_roi, savedir, subject, roi, betas_type='betas')
        save_betas(image_stims_unique, train_test_split_indexes, betas_roi_ave, savedir, subject, roi, betas_type='betas_ave')


def save_betas(img_stims, train_test_split_indexes, betas_roi, savedir, subject, roi, betas_type):
    betas_train = []
    betas_test = []

    # We want to have shared images across the subjects.
    for idx, img_stim_idx in enumerate(img_stims):
        if img_stim_idx in train_test_split_indexes:
            betas_test.append(betas_roi[idx,:])
        else:
            betas_train.append(betas_roi[idx,:])
            
    betas_train = np.stack(betas_train)
    betas_test = np.stack(betas_test)    
            
    np.save(f'{savedir}/{subject}_{roi}_{betas_type}_train.npy',betas_train)
    np.save(f'{savedir}/{subject}_{roi}_{betas_type}_test.npy',betas_test)


def save_img_stims(savedir, subject, image_stims_all, image_stims_unique, img_count, train_test_split_indexes):
    if not os.path.exists(f'{savedir}/{subject}_stims.npy'):           
        np.save(f'{savedir}/{subject}_stims.npy', image_stims_all)
        np.save(f'{savedir}/{subject}_stims_ave.npy', image_stims_unique)
        
    if img_count > 0:
        img_map_unique = np.repeat(True, repeats= img_count)
        temp = np.repeat(False, repeats= len(image_stims_unique) - img_count)
        img_map_unique = np.concatenate((img_map_unique, temp))
        img_map_all = np.isin(image_stims_all, image_stims_unique[:img_count])
    else:
        img_map_unique = np.repeat(True, repeats=len(image_stims_unique))
        img_map_all = np.repeat(True, repeats=len(image_stims_all))

    np.save(f'{savedir}/{subject}_stims_map.npy', img_map_all)
    np.save(f'{savedir}/{subject}_stims_ave_map.npy', img_map_unique)
    
    img_stims_included_in_experiment = image_stims_unique[img_map_unique]
    img_stims_for_testing = img_stims_included_in_experiment[np.isin(img_stims_included_in_experiment, train_test_split_indexes)]
    
    np.save(f'{savedir}/{subject}_stims_test_ids.npy', img_stims_for_testing)
    np.save(f'{savedir}/{subject}_stims_all_split_ids.npy', train_test_split_indexes)

if __name__ == "__main__":
    main()