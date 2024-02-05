import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    
    return parser.parse_args()


def main():
    sys.stdout.flush()


    args = get_args()
    target = args.target
    roi = args.roi
    subject = args.subject

    torch.cuda.manual_seed(42)
    # set_backend("torch_cuda")

    backend = set_backend("numpy")
    
    if target == 'text-latent' or target == 'image-latent':
        alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
        
    # if target == 'text-latent' or target == 'image-latent':
    #     alpha = [0.01, 0.1, 1]
        
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alpha),
        verbose = True
    )    
    
    mridir = f'../nsd-results/mrifeat/{subject}/'
    featdir = '../nsd-results/latents-320/subjfeat/'
    savedir = f'../nsd-results/latents-320/final/{subject}/'
    os.makedirs(savedir, exist_ok=True)
 
    img_stims_unique_test_map, img_stims_all_train_map = get_train_test_maps(mridir, args.subject)
    
    X_train = []
    X_test = []
    
    for croi in roi:
        cX_train = np.load(f'{mridir}/{subject}_{croi}_betas_train.npy').astype("float32")
        print(f'CxTrain: {cX_train.shape}')
        cX_train = cX_train[img_stims_all_train_map]
        print(f'CxTrain: {cX_train.shape}')
        
        cX_test = np.load(f'{mridir}/{subject}_{croi}_betas_ave_test.npy').astype("float32")
        print(f'CxTest: {cX_test.shape}')
        cX_test = cX_test[img_stims_unique_test_map]
        print(f'CxTest: {cX_test.shape}')
        
        X_train.append(cX_train)
        X_test.append(cX_test)
        
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
        
    Y_train = np.load(f'{featdir}/{subject}_each_{target}_train.npy').astype("float32").reshape([X_train.shape[0],-1])
    Y_test = np.load(f'{featdir}/{subject}_ave_{target}_test.npy').astype("float32").reshape([X_test.shape[0],-1])
        
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X_train {X_train.shape}, Y_train {Y_train.shape}, X_test {X_test.shape}, Y_test {Y_test.shape}')
    
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    pipeline.fit(X_train, Y_train)
    
    scores = pipeline.predict(X_test)       
    rs = correlation_score(Y_test.T, scores.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')
    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy',scores)
    
    # Should be close to 100%.
    scores2 = pipeline.predict(X_train)
    rs2 = correlation_score(Y_train.T, scores2.T)
    print(f'Accuracy on train dataset: {np.mean(rs2):3.3}')
    
    # scores = pipeline.predict(X_test)       
    # rs = correlation_score(Y_test.T, scores.T)
    # print(f'Prediction accuracy is: {np.mean(rs.detach().cpu().numpy()):3.3}')
    # np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy',scores.detach().cpu().numpy())
    
    # # # Should be close to 100%.
    # scores2 = pipeline.predict(X_train)
    # rs2 = correlation_score(Y_train.T, scores2.T)
    # print(f'Accuracy on train dataset: {np.mean(rs2.detach().cpu().numpy()):3.3}')


def get_train_test_maps(mridir, subject):
    img_stims_test_indexes = np.load(f'{mridir}/{subject}_stims_test_ids.npy')
    img_stims_all_split_indexes = np.load(f'{mridir}/{subject}_stims_all_split_ids.npy')

    img_stims_unique = np.load(f'{mridir}/{subject}_stims_ave.npy')
    img_stims_unique_test = img_stims_unique[np.isin(img_stims_unique, img_stims_all_split_indexes)]
    img_stims_unique_test_map = np.isin(img_stims_unique_test, img_stims_test_indexes)
        
    img_stims_unique_map = np.load(f'{mridir}/{subject}_stims_ave_map.npy')
    img_stims_unique2 = img_stims_unique[img_stims_unique_map]
    
    img_stims_all = np.load(f'{mridir}/{subject}_stims.npy')
    img_stims_all_train = img_stims_all[~np.isin(img_stims_all, img_stims_all_split_indexes)]
    img_stims_all_train_map = np.isin(img_stims_all_train, img_stims_unique2)
    
    return (img_stims_unique_test_map, img_stims_all_train_map)

if __name__ == "__main__":
    main()