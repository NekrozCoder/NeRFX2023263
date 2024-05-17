
import os, sys, cv2
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

    
def get_rnd(all_rays,all_idx_inside,num_ray):
    
    
    all_ = []
    chunk_per_im = num_ray//len(all_idx_inside)
    for i in range(len(all_idx_inside)):
        
            idx_inside  = all_idx_inside[i][np.random.choice(len(all_idx_inside[i]), size=[chunk_per_im], replace=False)] 
            all_.append(all_rays[i][idx_inside])
            
    return torch.cat(all_,0)     


def get_mask_idx(path_to_mask,num_im,i_train):
    
    mask_ = []
    for img_i in range(num_im):
       
        im = imageio.imread(os.path.join(path_to_mask, 'img_%0.3d.png'%(img_i)))
        im = np.float32(im)/255.0
        im = im.reshape([-1])
        inside = np.where(im== 1.0)[0]
        mask_.append(inside)

    
    mask = [mask_[i] for i in i_train]
    
    return mask


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    embedded = embed_fn(inputs)

    if viewdirs is not None:
        embedded_dirs = embeddirs_fn(viewdirs)

        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def run_IOR_network(inputs, fn, embed_fn, netchunk=1024*64):

    embedded = embed_fn(inputs)

 
    outputs = batchify(fn, netchunk)(embedded)

    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret



def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_models(args):
    """Instantiate NeRF and IOR MLP models.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 0
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)


    model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    

    
    embed_fn_ior, input_ch_ior = get_embedder(args.multires_views_ior, args.i_embed)
    model_ior = MLP_IOR(input_ch = input_ch_ior,D=args.netdepth_ior, W= args.netwidth_ior ,skips=[3]).to(device)
    model_ior.apply(init_weights)
    
    
    # only optimizing for the IoR field 
    grad_vars = list(model_ior.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    
    network_query_fn_ior = lambda inputs, network_fn : run_IOR_network(inputs, network_fn,
                                                                embed_fn = embed_fn_ior,
                                                                netchunk=args.netchunk)
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname


    ckpts = [os.path.join(basedir, expname,args.model_path, f) for f in sorted(os.listdir(os.path.join(basedir, expname,args.model_path))) if 'tar' in f]
    print('Trainined model path:',os.path.join(basedir, expname,args.model_path))
   
    
    print('Found ckpts', ckpts)
   
    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        print('Reloading the weights for the NeRF MLPs')

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
   
        if "network_ior_state_dict" in ckpt and not args.no_reload:
            model_ior.load_state_dict(ckpt['network_ior_state_dict'])
            print('Reloading the weights for the IoR MLP')

   

    ##########################

    render_kwargs_train = {
        
        'network_query_fn' : network_query_fn,
        'network_query_fn_ior': network_query_fn_ior,
        'perturb' : args.perturb,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'network_ior': model_ior,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



class IoR_emissionAbsorptionODE(nn.Module):

    def __init__(self,voxel_grid,query_fn,model,n_samples,step_size,bb_vals,scene_bound):
        super(IoR_emissionAbsorptionODE, self).__init__()

        self.voxel_grid = voxel_grid
        self.model = model
        self.query_fn = query_fn
        self.n_samples = n_samples
        self.step_size = step_size
        self.bounding_box_vals = bb_vals
        self.scene_bound = scene_bound
        
    def forward(self,t,y):
        
        
        pts = y[:,0:3]
        ray_dir = y[:,3:6]
        transmition_ = y[:,9:10]
        
        # query the 3D grid with trilinear interpolation
        raw = trilinear_interpolation(self.voxel_grid,pts,self.scene_bound)
        rgb = raw[...,:3]
        density = raw[...,3:]
        
        
        # determining whether a point is inside the bounding box or not  
        weights = get_bb_weights(pts,self.bounding_box_vals)


        # finding the points inside the bounding box
        insides = torch.where((weights<1.))[0]
        ior_grad = torch.zeros_like(pts)


        # setting the density to zero for the points inside the bounding box
        density= density*weights
        
        # calculating the gradient of IoR only for the points inside the bounding box
        if len(insides) > 0:
            pts_inside = pts[insides,:]
          
            with torch.enable_grad():
                
                dn = torch.autograd.functional.vjp(lambda x : self.query_fn(x,self.model), pts_inside ,v = torch.ones_like(pts_inside[:,0:1]),create_graph =True )                      
                ior_grad[insides,:] = dn[1]*(1.-weights[insides,:])
          
    
    
        dv_ds = self.step_size*ior_grad
        dx_ds = self.step_size*normalizing(ray_dir)
        alpha = 1. - torch.exp(-density*self.step_size/self.n_samples)
        alpha = alpha.clip(0.,1.)
        dc_dt = transmition_*rgb*alpha*self.n_samples
        dT_dt = -transmition_*alpha*self.n_samples
        dy_dt = torch.cat([dx_ds,dv_ds,dc_dt,dT_dt],-1)


        return dy_dt
    
    
def render_rays(ray_batch,
                voxel_grid,
                N_samples,
                network_query_fn_ior,
                bb_vals,
                network_ior,
                scene_bound,
                network_fn=None,
                network_query_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
   
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    
        
    pts =  rays_o + normalizing(rays_d)*near
    viewdir = normalizing(rays_d)
    color_ = torch.zeros_like(pts)
    transmition_ = torch.ones((N_rays,1))
  
    
    y0 = torch.cat((pts,viewdir,color_,transmition_),-1).to(device) 
    step_size =  far[0]-near[0]
    output = odeint(IoR_emissionAbsorptionODE(voxel_grid,network_query_fn_ior,network_ior,N_samples,step_size,bb_vals,scene_bound), y0, t_vals,method='euler')
    
    pts = output[:,:,0:3].permute(1,0,2)
   
    rgb_map = output[-1,:,6:9]
    

    ret = {'rgb_map' : rgb_map}
    if retraw:
         ret['raw'] = pts

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", default='glass_ball_new',type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='data/glass_ball', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*32, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", default=False,action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", default=True, action='store_true', 
                        help='do not reload the weights for the IoR MLP')
    parser.add_argument("--model_path", type=str, default='model_weights', 
                        help='path to trained model weights')


    # IoR model parameters 
    
    parser.add_argument("--netdepth_ior", type=int, default=6, 
                        help='layers in the IoR network')
    parser.add_argument("--netwidth_ior", type=int, default=64, 
                        help='channels per layer')
    parser.add_argument("--multires_views_ior", type=int, default=5, 
                        help='log2 of max freq for positional encoding (2D direction)')
    
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=128, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int,
                        help='Not used in this code')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", default=True,action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_mask", default=False,action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=1, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", default = True,action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", default = True, action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=10, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=50, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=500, 
                        help='frequency of weight ckpt saving')


    return parser

    

def train():

    parser = config_parser()
    args = parser.parse_args()
    
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        i_test = []
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]
    
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
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
            
            
        if args.spherify:
            far = np.minimum(far,2.0)    
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
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    
    
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_models(args)
    start = 0
    global_step = start
    
    
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    
    print('ray batching')
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [len(images),-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    rays_rgb = [rays_rgb[i] for i in i_train] # train images only
    
    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)
    
    
    print('loading mask images')
    path_mask = os.path.join(basedir, expname, 'masked_regions')
    inside_idx = get_mask_idx(path_mask,len(poses),i_train) # concatenating the indices of masked region in each view 
    
    
    print('calculating the scene bounds for 3D grid')
    scene_bound = get_scene_bound(near,far,H,W,K,poses,min_=.01,max_=.01)
    
    render_kwargs_test['scene_bound'] = scene_bound
    render_kwargs_train['scene_bound'] = scene_bound
    
    
    print('loading bounding box values')
    bounding_box_vals = np.load(os.path.join(basedir, expname, 'bounding_box/bounding_box_vals.npy'))
    
    
    render_kwargs_test['bb_vals'] = bounding_box_vals
    render_kwargs_train['bb_vals'] = bounding_box_vals
    
    
    print('fitting a 3D grid to learnt radiance field (Nerf)')
    query_function = render_kwargs_test['network_query_fn']
    nerf_coarse_model = render_kwargs_test['network_fn']
    Grid_res = 128
    
    reuse = False
    testimgdir = os.path.join(basedir, expname, 'voxel_grid.npy')
    
    skip_im = 4
    if reuse:
        voxel_grid = torch.tensor(np.load(testimgdir)).to(device)
    else:
        voxel_grid = get_voxel_grid(Grid_res,scene_bound,poses[0:-1:skip_im],query_function,nerf_coarse_model,False,bounding_box_vals)
        np.save(testimgdir,voxel_grid.cpu().numpy())
    
    
    
    N_iters = 5000 + 1 # number of total iteration
    iter_level = 1000 # number of iteration per each level
    sigma = 0.08 # initial sigma for the 3D Gaussian blur kernel
    N_rand = args.N_rand
    print('Begin')
    print('TEST views are', i_test)
    
    start = start + 0
    loss_vals = []
    
    
    for i in trange(start,N_iters):
        time0 = time.time()
    
        
        # taking random rays overl all views  
        batch = get_rnd(rays_rgb,inside_idx,N_rand)
        batch = torch.transpose(batch, 0, 1)
    
        batch_rays, target_s = batch[:2], batch[2]
        
        
        # creating a 3D Gaussian blur kernel
        filter_3d = lowpass_3d(Grid_res,sigma*2**(i//iter_level))
        
        # smoothing the grid with the 3D Gaussian blur kernel in Fourier domain
        voxel_grid_blur = voxel_lowpass_filtering(voxel_grid,filter_3d)
     
        render_kwargs_train['voxel_grid'] = voxel_grid_blur
        render_kwargs_train['N_samples'] = 128 
    
      
        #####  Core optimization loop  #####
        rgb, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=False,
                                                **render_kwargs_train)
    
        loss = img2mse(rgb, target_s)
        psnr = mse2psnr(loss)
    
    
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
       
        decay_rate = 0.1
        new_lrate = args.lrate* (decay_rate ** (global_step / N_iters))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
    
        dt = time.time()-time0
       
        loss_vals.append(loss.detach().cpu().numpy())
    
        
        if i%args.i_weights==0:
                model_path_dir = os.path.join(basedir, expname,args.model_path)
                os.makedirs(model_path_dir, exist_ok=True)
                path = os.path.join(basedir, expname,args.model_path, 'weights.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'network_ior_state_dict': render_kwargs_train['network_ior'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
    
    
    
        if i%args.i_img==0:
                
                render_kwargs_test['voxel_grid'] = voxel_grid  # rendering images with finest version of the grid
                img_i= i_test[5]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb ,extras = render(H, W,K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                
            
                testimgdir = os.path.join(basedir, expname, 'training_ior')
                os.makedirs(testimgdir, exist_ok=True)                    
                imageio.imwrite(os.path.join(testimgdir, 'rendered_{:03d}.png'.format(i)), to8b(rgb.cpu().numpy()))
                imageio.imwrite(os.path.join(testimgdir, 'ref_{:03d}.png'.format(img_i)), to8b(images[img_i]))
    
                # plt.plot(loss_vals)
                # plt.show()
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
       
    
        global_step += 1
    

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()    