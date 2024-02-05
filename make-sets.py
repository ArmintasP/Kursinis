import numpy as np
from tqdm import tqdm
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    topdir = '../nsd-results/latents-320/'
    mrifeatdir = '../nsd-results/mrifeat/'
    savedir = f'{topdir}/subjfeat/'
    featdir = f'{topdir}/{args.featname}/'
    
    img_stims_test_indexes = np.load(f'{mrifeatdir}/{args.subject}/{args.subject}_stims_test_ids.npy')
    
    if args.use_stim == 'ave':
        img_stims = np.load(f'{mrifeatdir}{args.subject}/{args.subject}_stims_ave.npy')
        img_stims_map = np.load(f'{mrifeatdir}/{args.subject}/{args.subject}_stims_ave_map.npy')
    else:
        img_stims = np.load(f'{mrifeatdir}{args.subject}/{args.subject}_stims.npy')
        img_stims_map = np.load(f'{mrifeatdir}/{args.subject}/{args.subject}_stims_map.npy')

    img_stims_in_experiment = img_stims[img_stims_map]
    
    features = []

    for _, img_stim_id in tqdm(enumerate(img_stims_in_experiment)):
        feature = np.load(f'{featdir}/{img_stim_id:06}.npy')
        features.append(feature)

    features = np.stack(features)    

    os.makedirs(savedir, exist_ok=True)

    img_stims_test_map = np.isin(img_stims_in_experiment, img_stims_test_indexes)

    features_train = features[img_stims_test_map == 0, :]
    features_test = features[img_stims_test_map == 1, :]

    np.save(f'{savedir}/{args.subject}_{args.use_stim}_{args.featname}_train.npy',features_train)
    np.save(f'{savedir}/{args.subject}_{args.use_stim}_{args.featname}_test.npy',features_test)


if __name__ == "__main__":
    main()