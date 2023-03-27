# @Author  : Junting Dong
# @Mail    : jtdong@zju.edu.cn
import torch
import numpy as np

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


def judge_bounds_inter(pts, batch):
    # 判断点在那个part的bound内
    # body: 0, face: 1, handl: 2, handr: 3, body-face: 4, body-handl: 5, body-handr: 6
    bound_id = torch.zeros((pts.shape[0], 1))
    face_ind = get_inside(pts, batch['tbounds_face'][0])
    handl_ind = get_inside(pts, batch['tbounds_handl'][0])
    handr_ind = get_inside(pts, batch['tbounds_handr'][0])
    body_ind = get_inside(pts, batch['tbounds_body'][0])

    body_face_ind = body_ind * face_ind
    body_handl_ind = body_ind * handl_ind
    body_handr_ind = body_ind * handr_ind

    face_only_ind = face_ind * ~body_face_ind
    handl_only_ind = handl_ind * ~body_handl_ind
    handr_only_ind = handr_ind * ~body_handr_ind
    body_only_ind = ~face_ind * ~handl_ind * ~handr_ind

    bound_id[body_only_ind] = 0
    bound_id[face_only_ind] = 1
    bound_id[handl_only_ind] = 2
    bound_id[handr_only_ind] = 3
    bound_id[body_face_ind] = 4
    bound_id[body_handl_ind] = 5
    bound_id[body_handr_ind] = 6

    return bound_id.squeeze()

def judge_bounds(pts, batch):
    # 判断点在那个part的bound内
    # background: -1, body: 0, face: 1, handl: 2, handr: 3
    bound_id = torch.zeros((pts.shape[0], 1))
    face_only_ind = get_inside(pts, batch['tbounds_face'][0])
    handl_only_ind = get_inside(pts, batch['tbounds_handl'][0])
    handr_only_ind = get_inside(pts, batch['tbounds_handr'][0])
    body_ind = get_inside(pts, batch['tbounds_body'][0])
    body_only_ind = body_ind * ~face_only_ind * ~handl_only_ind * ~handr_only_ind
    bg_only_ind = ~body_only_ind * ~face_only_ind * ~handl_only_ind * ~handr_only_ind
    bound_id[bg_only_ind] = -1
    bound_id[body_only_ind] = 0
    bound_id[face_only_ind] = 1
    bound_id[handl_only_ind] = 2
    bound_id[handr_only_ind] = 3
    return bound_id.squeeze()



def PerPartCompute(pts, batch, funcs, func_args):

    bound_id = judge_bounds(pts, batch)
    num_part = int(bound_id.max().item() + 1)
    args_index = func_args['index']
    args_noindex = func_args['noindex']
    args_step = func_args['step'] if 'step' in func_args.keys() else {}
    args_meta = func_args['meta'] if 'meta' in func_args.keys() else {}
    if 'init' in args_meta.keys():
        init = args_meta['init']
    else:
        init = 0
    output = None
    for part_i in range(-1, num_part):
        part_i_index = (bound_id == part_i)
        if part_i_index.sum() == 0:
            continue
        if len(part_i_index.shape) == 0:
            part_i_index = part_i_index[None]
        func_i = funcs[part_i]
        args_i = {key: val[part_i_index] for key,val in args_index.items()}
        args_i.update(args_noindex)
        if 'step' in func_args.keys():
            args_step_i = {key: val[part_i+1] for key, val in args_step.items()}
            args_i.update(args_step_i)
        output_i = func_i(**args_i)
        if output is None:
            out_dim = output_i.shape[-1]
            output = torch.ones([pts.shape[0], out_dim]).to(pts) * init
        output[part_i_index] = output_i

    if output is None:
        #说明所有点都在bg里
        part_i = -1
        part_i_index = (bound_id == part_i)
        if len(part_i_index.shape) == 0:
            part_i_index = part_i_index[None]
        func_i = funcs[0]
        args_i = {key: val[part_i_index] for key,val in args_index.items()}
        args_i.update(args_noindex)
        if 'step' in func_args.keys():
            args_step_i = {key: val[part_i] for key, val in args_step.items()}
            args_i.update(args_step_i)
        output_i = func_i(**args_i)
        out_dim = output_i.shape[-1]
        output = torch.ones([pts.shape[0], out_dim]).to(pts) * init

    return output

# 单元测试bg类

def judge_bounds_test(pts, batch):
    # 判断点在那个part的bound内
    # body: 0, face: 1, handl: 2, handr: 3
    bound_id = torch.zeros((pts.shape[0], 1))
    face_only_ind = get_inside(pts, batch['tbounds_face'][0])
    handl_only_ind = get_inside(pts, batch['tbounds_handl'][0])
    handr_only_ind = get_inside(pts, batch['tbounds_handr'][0])
    body_only_ind = ~face_only_ind * ~handl_only_ind * ~handr_only_ind
    bound_id[body_only_ind] = 0
    bound_id[face_only_ind] = 1
    bound_id[handl_only_ind] = 2
    bound_id[handr_only_ind] = 3

    return bound_id.squeeze()


def PerPartCompute_test(pts, batch, funcs, func_args):

    bound_id = judge_bounds_test(pts, batch)
    num_part = int(bound_id.max().item() + 1)
    args_index = func_args['index']
    args_noindex = func_args['noindex']
    args_step = func_args['step'] if 'step' in func_args.keys() else {}
    output = None
    for part_i in range(num_part):
        part_i_index = (bound_id == part_i)
        if part_i_index.sum() == 0:
            continue
        if len(part_i_index.shape) == 0:
            part_i_index = part_i_index[None]
        func_i = funcs[part_i]
        args_i = {key: val[part_i_index] for key,val in args_index.items()}
        args_i.update(args_noindex)
        if 'step' in func_args.keys():
            args_i.update(args_step[part_i])
        output_i = func_i(**args_i)
        if output is None:
            out_dim = output_i.shape[-1]
            output = torch.zeros([pts.shape[0], out_dim]).to(pts)
        output[part_i_index] = output_i

    return output


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
    return near, far

def hand_prior_sample(skeleton):
    # skeleton: 52x3, SMPL-H
    l_index = np.arange(22, 22 + 15).reshape(-1, 3)
    r_index = np.arange(37, 37 + 15).reshape(-1, 3)
    center = []
    for index in [l_index, r_index]:
        l_kpts = skeleton[index]
        l_index_middle = (l_kpts[0] + l_kpts[1]) / 2
        l_middle_lring = (l_kpts[1] + l_kpts[3]) / 2
        l_lring_pinky = (l_kpts[3] + l_kpts[2]) / 2
        l_diff = np.stack([l_index_middle, l_middle_lring, l_lring_pinky])
        l_center = (l_diff[:, :-1] + l_diff[:, 1:]) / 2
        center.append(l_center)
    center = np.concatenate(center, axis=0)
    # random sample points for sdf training
    N_hprior = 100
    noise = np.random.rand(N_hprior, 1)
    points = center[:, :1] + (center[:, 1:] - center[:, :1]) * 3 * noise
    points[..., 1] += (np.random.rand(1, N_hprior) - 0.5) * 0.05
    points[..., 2] += (np.random.rand(1, N_hprior) - 0.5) * 0.002
    points = points.reshape(-1, 3)

    return points

names_smplh = [
    'Pelvis',
    'L_Hip',
    'R_Hip',
    'Spine1',
    'L_Knee',
    'R_Knee',
    'Spine2',
    'L_Ankle',
    'R_Ankle',
    'Spine3',
    'L_Foot',
    'R_Foot',
    'Neck',
    'L_Collar',
    'R_Collar',
    'Head',
    'L_Shoulder',
    'R_Shoulder',
    'L_Elbow',
    'R_Elbow',
    'L_Wrist',
    'R_Wrist',
    'lindex0',
    'lindex1',
    'lindex2',
    'lmiddle0',
    'lmiddle1',
    'lmiddle2',
    'lpinky0',
    'lpinky1',
    'lpinky2',
    'llring0',
    'llring1',
    'llring2',
    'lthumb0',
    'lthumb1',
    'lthumb2',
    'rindex0',
    'rindex1',
    'rindex2',
    'rmiddle0',
    'rmiddle1',
    'rmiddle2',
    'rpinky0',
    'rpinky1',
    'rpinky2',
    'rlring0',
    'rlring1',
    'rlring2',
    'rthumb0',
    'rthumb1',
    'rthumb2',
]
