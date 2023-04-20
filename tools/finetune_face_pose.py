# @Author  : Junting Dong
# @Mail    : jtdong@zju.edu.cn
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
import os.path as osp
import sys
import glob
from tqdm import tqdm
sys.path.insert(0, osp.join(os.path.dirname(__file__), '..'))
from lib.utils.renderer.renderAPI import vis_mesh, vis_mask
import numpy as np
import cv2
import os.path as osp
import openmesh
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from pytorch3d.ops import iterative_closest_point
import sys
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj, IO
import argparse
sys.path.insert(0, osp.join(os.path.dirname(__file__), '..'))


def np2mesh(mesh, xnp, path):
    mesh.points()[:] = xnp
    openmesh.write_mesh(path, mesh, binary=True)


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


def set_requires_nograd(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = False


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def rot_trans_pts(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans[:, :, None]
    return rott_geo.permute(0, 2, 1)


def forward_rott(geometry, euler_angle, trans):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    return rott_geo

def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K / K[2, 2]
    R_pred = R
    t_pred = -R.dot((t[:3] / t[3])[:, 0])
    P_pred = np.zeros((3, 4), dtype=np.float32)
    P_pred[:3, :3] = R_pred
    P_pred[:3, 3] = t_pred
    P_pred = K.dot(P_pred)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose

def load_cameras():
    id = 'colmap'
    cameras = np.load(osp.join(path, 'cameras_' + id + '.npz'))
    K_all = []
    RT_all = []
    K, Rt = load_K_Rt_from_P(cameras['world_mat_0'][:3])
    K_inv = np.linalg.inv(K)
    # camera intris
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    frame_num = len(cameras.files)
    sel_ids = np.arange(0, frame_num)
    for sel_id in sel_ids:
        # K, Rt = load_K_Rt_from_P(cameras['world_mat_%d' % sel_id][:3])
        # R = Rt[:3, :3]
        # T = Rt[:3, 3:]
        Rt_sfm = K_inv.dot(cameras['world_mat_%d' % sel_id])
        R_sfm = Rt_sfm[:3, :3]
        # s = np.sqrt((Rt_sfm @ Rt_sfm.T)[0, 0])
        s = np.sqrt((R_sfm @ R_sfm.T)[0, 0])
        R_sfm /= s
        T_sfm = Rt_sfm[:3, 3:] / s
        Rt_sfm[:3, :3] = R_sfm
        Rt_sfm[:3, 3:] = T_sfm
        RT_all.append(Rt_sfm)
        # print(R_sfm@R_sfm.T)
        # cameras_sfm.append(K_inv.dot(cameras['world_mat_%d' % sel_id]))

    # camera extri : translate from world space to camera space
    Rts = np.stack(RT_all)
    # K = np.stack(K_all)
    cams = {
        'K': K,
        'RT': Rts
    }
    np.save(osp.join(path, 'cameras'), cams)

def register_flame2smplh():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    root_path = args.path
    body_path = args.bodypath
    source_mesh = IO().load_mesh(osp.join(root_path, 'tpose_mesh.ply'))
    target_mesh = IO().load_mesh(osp.join(body_path, 'tpose_mesh.ply'))

    gt_pts_ori = target_mesh.verts_list()[0].float().cuda()
    gt_normals_ori = target_mesh.verts_normals_list()[0].float().cuda()
    pred_pts_ori = source_mesh.verts_list()[0].float().cuda().unsqueeze(0)

    # select face pts
    # register_bound_reduce = 0.08
    # register_upbound_reduce = 0.06
    # register_forwardbound_reduce = 0# 0.12
    gt_pts_bound = get_face_bounds(gt_pts_ori.cpu().numpy())
    gt_pts_bound[0, 1] += register_bound_reduce#0.13
    gt_pts_bound[1, 1] -= register_upbound_reduce#0.13
    gt_pts_bound[0, 2] += register_forwardbound_reduce#0.13
    inside_idx = get_inside(gt_pts_ori.cpu().numpy(), gt_pts_bound)
    gt_pts = gt_pts_ori[inside_idx]
    gt_normals = gt_normals_ori[inside_idx]
    pred_pts_bound = get_face_bounds(pred_pts_ori[0].cpu().numpy())
    pred_pts_bound[0, 1] += register_bound_reduce#0.13
    pred_pts_bound[1, 1] -= register_upbound_reduce#0.13
    pred_pts_bound[0, 2] += register_forwardbound_reduce#0.13

    inside_idx = get_inside(pred_pts_ori[0].cpu().numpy(), pred_pts_bound)
    pred_pts = pred_pts_ori[0, inside_idx]
    # from lib.utils.debugger import dbg
    # dbg.showL3D([gt_pts.cpu()[::20], pred_pts.cpu()[::20]])
    # import ipdb; ipdb.set_trace(context=11)
    if pred_pts.shape[0] > 40000:
        downsample_scale = pred_pts.shape[0] // 40000
        pred_pts = pred_pts[::downsample_scale, :]
    euler_angle = gt_pts.new_zeros((1, 3), requires_grad=True)
    trans = gt_pts.new_zeros((1, 3), requires_grad=True)
    scale = gt_pts.new_ones((1, 1), requires_grad=True)

    set_requires_grad([euler_angle, trans])
    optimizer_rigid = torch.optim.Adam([euler_angle, trans], lr=.01)
    # optimizer_rigid = torch.optim.Adam([euler_angle, trans, scale], lr=.01)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
        gt_pts.detach().cpu().numpy())
    for iter in range(150):
        pred_pts = pred_pts.reshape(1, -1, 3)
        geometry = forward_rott(pred_pts, euler_angle, trans)
        geometry = scale * geometry

        _, ids = nbrs.kneighbors(geometry[0, :, :].detach().cpu().numpy())

        gt_pair_pts = gt_pts[torch.as_tensor(ids).squeeze(), :]
        gt_pair_normals = gt_normals[torch.as_tensor(ids).squeeze(), :]
        nan = torch.isnan(gt_pair_normals).sum(dim=1)
        inf = torch.isinf(gt_pair_normals).sum(dim=1)
        valid_idx = torch.where((nan+inf)==0)[0]
        loss_pp = torch.mean(torch.abs(
            torch.sum(((geometry[0, :, :][valid_idx]-gt_pair_pts[valid_idx])*gt_pair_normals[valid_idx]), dim=1)))

        loss = loss_pp
        if iter > 1000:
            loss = loss_pp
        if iter == 600:
            optimizer_rigid = torch.optim.Adam(
                [euler_angle, trans, scale], lr=.001)
        if iter % 50 == 0:
            print('iter: {}, scale: {}, loss_pp: {}'.format(iter, scale.item(), loss_pp.item()))
        optimizer_rigid.zero_grad()
        loss.backward()
        optimizer_rigid.step()
    geometry = forward_rott(pred_pts_ori, euler_angle, trans)
    source_mesh.verts_list()[0] = geometry[0].detach().cpu()
    IO().save_mesh(source_mesh, os.path.join(root_path, 'regist_face.ply'))

    R_f = euler2rot(euler_angle).detach().cpu().numpy()
    T_f = trans.detach().cpu().numpy()
    s = scale.detach().cpu().numpy()
    transform = {
        'R_f': R_f,
        'T_f': T_f,
        's': s
    }
    np.save(osp.join(root_path, 'transform'), transform)

def get_face_bounds(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds

def get_inside(pts, bound):
    inside = pts > bound[:1]
    inside = inside * (pts < bound[1:])
    if isinstance(inside, torch.Tensor):
        inside = torch.sum(inside, dim=1) == 3
    elif isinstance(inside, np.ndarray):
        inside = np.sum(inside, axis=1) == 3
    else:
        raise ValueError

    return inside
def vis_register_face():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    mesh = load_objs_as_meshes([osp.join(path, 'regist_flame.obj')])
    vertices = mesh.verts_list()[0].numpy()
    faces = mesh.faces_list()[0].numpy()
    np.save(osp.join(path, 'tvertices_face'), vertices)
    np.save(osp.join(path, 'faces_face'), faces)
    cameras = np.load(osp.join(path, 'cameras.npy'), allow_pickle=True)
    transform = np.load(osp.join(path, 'transform.npy'), allow_pickle=True).item()
    R_f = transform['R_f']
    T_f = transform['T_f']
    s = transform['s']
    RT = cameras.item()['RT']
    K = cameras.item()['K']
    nf = RT.shape[0]
    imgs = sorted(glob.glob(osp.join(path, 'image', '*.jpg')))
    new_cams = []
    for i in tqdm(range(nf)):
        img = cv2.imread(imgs[i])
        cam_i = {'K': K,
                 'R': RT[i][:3, :3] @ R_f[0].T,
                 'T': RT[i][:3, 3:] * s - RT[i][:3, :3] @ R_f[0].T @ T_f.T * s}
        new_cams.append(cam_i)
        imgname = osp.join(path, 'render_register', osp.basename(imgs[i]))
        imgname_mask = osp.join(path, 'render_mask', osp.basename(imgs[i]))
        os.makedirs(osp.dirname(imgname), exist_ok=True)
        os.makedirs(osp.dirname(imgname_mask), exist_ok=True)
        vis_mesh(vertices, faces, cam_i, img, imgname)
        vis_mask(vertices, faces, cam_i, img, imgname_mask, add_back=False)
    np.save(osp.join(path, 'new_cameras'), new_cams)

def get_grid_points(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.02
    max_xyz += 0.02
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    vsize = 0.002
    voxel_size = [vsize, vsize, vsize]
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts

def get_distance():
    from psbody.mesh import Mesh

    vertices = np.load(osp.join(path, 'tvertices_face.npy'))
    faces = np.load(osp.join(path, 'faces_face.npy'))
    face_mesh = Mesh(vertices, faces)

    # create grid points in the pose space
    pts = get_grid_points(vertices)
    sh = pts.shape
    pts = pts.reshape(-1, 3)
    _, norm = face_mesh.closest_vertices(pts, use_cgal=True)
    norm = norm.reshape(*sh[:3], 1).astype(np.float32)
    np.save(osp.join(path, 'distance.npy'), norm)

def get_mask(use_mask='mask'):
    img_path = osp.join(path, 'image')
    mask_path = osp.join(path, 'mask_')
    mask_ori_path = osp.join(path, 'mask')
    maskimg_path = osp.join(path, 'maskimg')
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(maskimg_path, exist_ok=True)
    img_files = sorted(glob.glob(osp.join(img_path, '*.jpg')))
    for img_i in tqdm(img_files):
        img = cv2.imread(img_i) / 255
        img_msk = (np.array(img).mean(axis=-1) != 1).astype(np.float32)
        ori_msk = img_i.replace('image', use_mask).replace('jpg', 'png')
        ori_msk = cv2.imread(ori_msk) / 255
        msk_combine = img_msk * ori_msk[..., 0]
        cv2.imwrite(osp.join(mask_path, osp.basename(img_i)), msk_combine)
        cv2.imwrite(osp.join(maskimg_path, osp.basename(img_i)), msk_combine[..., None]*img*255)
        # import matplotlib.pylab as plt;plt.figure();plt.imshow(orig_msk);plt.show()
        # from lib.utils.debugger import dbg
        # dbg.showL2D([orig_msk, msk, orig_msk*msk])
        # import ipdb; ipdb.set_trace(context=11)

def reform():
    out = osp.join(path, 'output')
    os.makedirs(out, exist_ok=True)
    out_img = osp.join(out, 'images')
    os.makedirs(out_img, exist_ok=True)
    out_msk = osp.join(out, 'masks')
    os.makedirs(out_msk, exist_ok=True)
    out_render_msk = osp.join(out, 'render_masks')
    os.makedirs(out_render_msk, exist_ok=True)
    # deal with images
    os.system('cp {}/*.jpg {}/'.format(osp.join(path, 'image'), out_img))
    # deal with masks
    os.system('cp {}/* {}/'.format(osp.join(path, 'mask_'), out_msk))
    os.system('cp {}/* {}/'.format(osp.join(path, 'render_mask'), out_render_msk))
    os.system('cp {}/distance.npy {}/'.format(path, out))
    os.system('cp {}/faces_face.npy {}/'.format(path, out))
    os.system('cp {}/new_cameras.npy {}/'.format(path, out))
    os.system('cp {}/tvertices_face.npy {}/'.format(path, out))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--step', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--bodypath', type=str)
    parser.add_argument('--register_bound_reduce', type=float, default=0.08)
    parser.add_argument('--register_upbound_reduce', type=float, default=0.06)
    parser.add_argument('--register_forwardbound_reduce', type=float, default=0.0)
    args = parser.parse_args()
    path = args.path

    register_bound_reduce = args.register_bound_reduce
    register_upbound_reduce = args.register_upbound_reduce
    register_forwardbound_reduce = args.register_forwardbound_reduce


    if args.debug:
        steps = ['step2']
        # steps = ['step1', 'step4', 'step6']
        # steps = ['step1', 'step3', 'step4', 'step5']
    elif args.step == 'step3':
        steps = ['step3', 'step6']
    else:
        steps = ['step1', 'step2', 'step4', 'step5']

    if 'step1' in steps:
        # STEP1: load cameras
        print('deal with STEP1')
        load_cameras()

    if 'step2' in steps:
        # STEP2: register flame to smplh
        print('deal with STEP2')
        register_flame2smplh()

    if 'step3' in steps:
        # STEP3: get distance of flame
        print('deal with STEP3')
        get_distance()

    if 'step4' in steps:
        # STEP4: vis registered mesh
        print('deal with STEP4')
        vis_register_face()

    if 'step5' in steps:
        # STEP5: get mask
        print('deal with STEP5')
        get_mask('matting')



    if 'step6' in steps:
        # STEP6: reform the output file
        print('deal with STEP6')
        reform()


