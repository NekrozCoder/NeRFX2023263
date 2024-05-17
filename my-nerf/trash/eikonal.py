import os
import cv2
import time
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from data_loader.load_llff import load_llff_data
from data_loader.load_blender import load_blender_data
from model import *
from render import *


def get_bb_weights(pts,bounding_box_val,beta=200):
    
    center  = bounding_box_val[0] 
    rad = bounding_box_val[1]

    x_dist = torch.abs(pts[...,0:1] - torch.tensor(center[0]).to(pts))
    y_dist = torch.abs(pts[...,1:2] - torch.tensor(center[1]).to(pts))
    z_dist = torch.abs(pts[...,2:3] - torch.tensor(center[2]).to(pts))

    weights = torch.sigmoid(beta*(rad[0]-x_dist))*torch.sigmoid(beta*(rad[1]-y_dist))*torch.sigmoid(beta*(rad[2]-z_dist))

    return 1.0 - weights

def find_box(args):
    
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]
    
        if args.llffhold > 0:
            print('number of images to annotate: ', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
    
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
    
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    
    
     # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    
    if args.render_test:
        render_poses = np.array(poses[i_test])
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    # os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # f = os.path.join(basedir, expname, 'args.txt')
    # with open(f, 'w') as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write('{} = {}\n'.format(arg, attr))
    # if args.config is not None:
    #     f = os.path.join(basedir, expname, 'config.txt')
    #     with open(f, 'w') as file:
    #         file.write(open(args.config, 'r').read())
    
     # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_test.update(bds_dict)
    
    render_poses = torch.Tensor(render_poses).to(device)
    N_rand = args.N_rand
    use_batching = not args.no_batching
    
    poses = torch.Tensor(poses).to(device)
    
    if not args.reload_bb:
        point_matrix = []
         
        counter = 0
        def mousePoints(event,x,y,flags,params):
            if event == cv2.EVENT_LBUTTONDOWN:
                point_matrix.append([x,y])
                cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        
        
        num_ = len(i_test) #len(poses)//10
        cam_ind = i_test# np.arange(0,len(poses),num_)
        counter = 0
               
        img = to8b(images[cam_ind[counter],:,:,3::-1])
        rays_o, rays_d = get_rays(H, W, K, poses[cam_ind[counter]])
        rays = torch.stack([rays_o, rays_d],0)
        all_rays = []
        
        text = "Please press A to continue to the next image "
        coordinates = (10,20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        color = (0,0,255)
        thickness = 0
                
        while True:
            
            img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow(" Image ", img)
            
            cv2.setMouseCallback(" Image ", mousePoints)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            
            if key == ord('a'):
                counter = counter+1
                pts_inside = np.array(point_matrix)
                all_rays.append(rays[:,pts_inside[:,1],pts_inside[:,0],:])
                if counter == num_:
                    break
                rays_o, rays_d = get_rays(H, W, K, poses[cam_ind[counter]])
                rays = torch.stack([rays_o, rays_d],0)
                point_matrix = []
                img = to8b(images[cam_ind[counter],:,:,3::-1])
                
        cv2.destroyAllWindows()
        
        all_rays = torch.cat(all_rays,1)
        
        print('Started finding the transparent object bounding box')
        render_kwargs_test['bb_vals'] = []
        render_kwargs_test['masking'] = False
        render_kwargs_test['N_importance'] = 256
        
        with torch.no_grad():
                _, _, _,weights, _ = render(H, W,K, chunk=args.chunk, rays= all_rays,
                                                    **render_kwargs_test)
                
        point_cloud = weights.cpu().numpy()
        
        
        max_pts = np.percentile((point_cloud),100 - args.quantile,axis=0)
        min_pts = np.percentile((point_cloud),args.quantile,axis=0)
        center = 0.5*(min_pts+max_pts)
        radi = max_pts-center
        radi = args.ratio*radi
        
        bounding_box_vals = np.stack((center,radi),0)
        
        path_bounding_box = os.path.join(basedir, expname,"bounding_box")
        os.makedirs(path_bounding_box, exist_ok=True)
        path = os.path.join(path_bounding_box, 'bounding_box_vals.npy')
        np.save(path,bounding_box_vals)
        print('Finised finding the transparent object bounding box')
        
        
    else:
            
        path_bounding_box = os.path.join(basedir, expname,"bounding_box")
        path = os.path.join(path_bounding_box, 'bounding_box_vals.npy')
        bounding_box_vals = np.load(path)

            
        
    
    render_kwargs_test['bb_vals'] = bounding_box_vals
    render_kwargs_test['masking'] = True
    render_kwargs_test['N_importance'] = 64
    
    
    
    print('Rendering a test image with and without transparent object')
    
    path_bounding_box = os.path.join(basedir, expname,"bounding_box")

    img_i = i_test[5]
    
    with torch.no_grad():
            rgb_gt_without, _,_,_,_ = render(H, W,K, chunk=args.chunk, c2w= poses[img_i],
                                                **render_kwargs_test)
      
    
    imageio.imwrite(os.path.join(path_bounding_box, 'without_{:03d}.png'.format(img_i)), to8b(rgb_gt_without.cpu().numpy()))
        
    render_kwargs_test['masking'] = False

    with torch.no_grad():
            rgb_gt_with, _,_,_,_ = render(H, W,K, chunk=args.chunk, c2w= poses[img_i],
                                                **render_kwargs_test)
          
    imageio.imwrite(os.path.join(path_bounding_box, 'with_{:03d}.png'.format(img_i)), to8b(rgb_gt_with.cpu().numpy()))
        
    print('Done!')
    print('Saved in the folder \"bounding_box\"')
    
    
    print('Finding the region in each image crossing the 3D bounding box')
        
    path_to_mask_out = os.path.join(basedir, expname, 'masked_regions')
    os.makedirs(path_to_mask_out, exist_ok=True) 
    
    
    for img_i in tqdm(range(len(poses))):
        
        pose = poses[img_i, :3,:4]  
        rays_o, rays_d = get_rays(H, W, K, pose)
        t_vals = torch.linspace(0., 1., steps=128)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts_ = rays_o[...,None,:] + rays_d[...,None,:]*z_vals[:,None]
        mask = 1. - get_bb_weights(pts_,bounding_box_vals) 
        diff = torch.sum(mask,2)>0.0
        imageio.imwrite(os.path.join(path_to_mask_out, 'img_%0.3d.png'%(img_i)), to8b(diff.cpu().numpy()))
   
    print('Done!')
    print('Saved in the folder \"masked_regions\"')