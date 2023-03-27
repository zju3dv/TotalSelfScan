import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh
import torch


def get_rays_within_bounds_test(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    if False:
        from lib.utils.debugger import dbg
        sample_pts_near = ray_o + ray_d * near[:, None]
        sample_pts_far = ray_o + ray_d * far[:, None]
        dbg.showL3D([bounds, sample_pts_near, sample_pts_far, ray_o])
        dbg.showL3D([bounds, ray_o, ray_o+ray_d])
        import ipdb; ipdb.set_trace(context=11)
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


def get_near_far_(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[(ray_d >= 0) * (ray_d < 1e-6)] = 1e-6
    ray_d[(ray_d > -1e-6) * (ray_d < 0)] = -1e-6
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-4
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split, all_in_mask=False):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    # import matplotlib.pylab as plt;plt.figure();plt.imshow(bound_mask);plt.show()

    if cfg.mask_bkgd:
        img[bound_mask != 1] = 0

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0

    # interhand mask
    interhand_msk = np.zeros_like(msk)
    left_top = np.argwhere(msk==1).min(axis=0)
    right_bottom = np.argwhere(msk==1).max(axis=0)
    interhand_msk[left_top[0]: right_bottom[0], left_top[1]: right_bottom[1]] = 1
    interhand_msk -= msk
    interhand_msk[interhand_msk!=1] = 0
    # import matplotlib.pylab as plt;plt.figure();plt.imshow(interhand_msk);plt.show()
    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        if all_in_mask:
            body_sample_ratio = 1
            interhand_sample_ratio = 0
        else:
            body_sample_ratio = cfg.body_sample_ratio
            interhand_sample_ratio = cfg.interhand_sample_ratio

        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_interhand = int((nrays - nsampled_rays) * interhand_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face - n_interhand

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]

            # sample rays outside the hand mask
            coord_interhand = np.argwhere(interhand_msk == 1)
            if len(coord_interhand) > 0:
                coord_interhand = coord_interhand[np.random.randint(
                    0, len(coord_interhand), n_interhand)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            elif len(coord_interhand) > 0:
                coord = np.concatenate([coord_body, coord_interhand, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(coord[:,1], coord[:,0],'r*');plt.show()

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_rays_within_bounds(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def sample_coord(img, msk, train_coord, K, R, T, bounds, nrays):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if cfg.mask_bkgd:
        img[bound_mask != 1] = 0

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0

    nsampled_rays = 0
    ray_o_list = []
    ray_d_list = []
    rgb_list = []
    near_list = []
    far_list = []
    coord_list = []

    num_train_coord = len(train_coord['coord'])

    while nsampled_rays < nrays:
        n_rand = nrays - nsampled_rays

        # sample coordinates
        inds = np.random.randint(0, num_train_coord, n_rand)
        coord = train_coord['coord'][inds]
        near = train_coord['near'][inds]
        far = train_coord['far'][inds]

        # select coordinates whose masks are not 100
        coord_msk = msk[coord[:, 0], coord[:, 1]]
        inds = coord_msk != 100
        coord = coord[inds]
        near = near[inds]
        far = far[inds]

        ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
        ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
        rgb_ = img[coord[:, 0], coord[:, 1]]

        ray_o_list.append(ray_o_)
        ray_d_list.append(ray_d_)
        rgb_list.append(rgb_)
        near_list.append(near)
        far_list.append(far)
        coord_list.append(coord)
        nsampled_rays += len(near)

    ray_o = np.concatenate(ray_o_list).astype(np.float32)
    ray_d = np.concatenate(ray_d_list).astype(np.float32)
    rgb = np.concatenate(rgb_list).astype(np.float32)
    near = np.concatenate(near_list).astype(np.float32)
    far = np.concatenate(far_list).astype(np.float32)
    coord = np.concatenate(coord_list)
    mask_at_box = np.ones([len(coord)]).astype(np.bool)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_smpl_data(ply_path):
    ply = trimesh.load(ply_path)
    xyz = np.array(ply.vertices)
    nxyz = np.array(ply.vertex_normals)

    if cfg.add_pointcloud:
        # add random points
        xyz_, ind_ = trimesh.sample.sample_surface_even(ply, 5000)
        nxyz_ = ply.face_normals[ind_]
        xyz = np.concatenate([xyz, xyz_], axis=0)
        nxyz = np.concatenate([nxyz, nxyz_], axis=0)

    xyz = xyz.astype(np.float32)
    nxyz = nxyz.astype(np.float32)

    return xyz, nxyz


def get_acc(coord, msk):
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def rotate_smpl(xyz, nxyz, t):
    """
    t: rotation angle
    """
    xyz = xyz.copy()
    nxyz = nxyz.copy()
    center = (np.min(xyz, axis=0) + np.max(xyz, axis=0)) / 2
    xyz = xyz - center
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R = R.astype(np.float32)
    xyz[:, :2] = np.dot(xyz[:, :2], R.T)
    xyz = xyz + center
    # nxyz[:, :2] = np.dot(nxyz[:, :2], R.T)
    return xyz, nxyz, center


def transform_can_smpl(xyz):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > cfg.rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    num_joints = poses.shape[0]
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([num_joints, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([num_joints, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    return transforms

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros([batch_size, 1], device=poses.device)
    K = torch.cat((zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros), dim=1)
    K = K.reshape([batch_size, 3, 3])
    ident = torch.eye(3, device=poses.device)[None]
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat

def get_rigid_transformation_torch_pose(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    num_pca = parents.shape[0]
    rot_mats = batch_rodrigues_torch(poses)
    # obtain the relative joints
    rel_joints = torch.zeros_like(joints)
    rel_joints[1:] = joints[1:] - joints[parents[1:]]
    rel_joints[0] = joints[0]
    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], axis=2)
    padding = torch.zeros([num_pca, 1, 4], device=transforms_mat.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.mm(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=0)

    # obtain the rigid transformation
    padding = torch.zeros([num_pca, 1], device=poses.device)
    joints_homogen = torch.cat([joints, padding], axis=1)
    rel_joints_ = torch.sum(transforms * joints_homogen[:, None], axis=2)
    # import ipdb; ipdb.set_trace(context=11)
    transforms_final = torch.zeros_like(transforms)
    transforms_final[..., :3] = transforms[..., :3]
    transforms_final[..., 3] = transforms[..., 3] - rel_joints_

    # transforms[..., 3] = transforms[..., 3].clone() - rel_joints_
    # transforms = transforms.astype(np.float32)

    return transforms_final


def padding_bbox(bbox, img):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size=80, max_size=88):
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    min_size = int(min(min_size, 0.8 * min_HW))
    H_size = np.random.randint(min_size, max_size)
    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coord = np.argwhere(msk == 1)
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bounds(xyz, part='body'):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    if part == 'body':
        dxyz = cfg.box_padding
    elif part == 'face':
        dxyz = 0.02
    elif part == 'hand':
        dxyz = 0.01
    elif part == 'multi':
        dxyz = 0
    elif part == 'mask':
        dxyz = 0.1
    min_xyz -= dxyz
    max_xyz += dxyz
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds

def get_face_bounds(xyz, delta=0.02):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= delta
    max_xyz += delta
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds

def prepare_sp_input(xyz):
    # obtain the bounds for coord construction
    bounds = get_bounds(xyz)
    # construct the coordinate
    dhw = xyz[:, [2, 1, 0]]
    min_dhw = bounds[0, [2, 1, 0]]
    max_dhw = bounds[1, [2, 1, 0]]
    voxel_size = np.array(cfg.voxel_size)
    coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
    # construct the output shape
    out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh, bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img
