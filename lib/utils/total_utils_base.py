# @Author  : Junting Dong
# @Mail    : jtdong@zju.edu.cn
import torch
import numpy as np

def get_inside(pts, bound):
    inside = pts > bound[:1]
    inside = inside * (pts < bound[1:])
    inside = torch.sum(inside, dim=1) == 3

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


def PerPartCompute(pts, batch, funcs, func_args):

    bound_id = judge_bounds(pts, batch)
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
            args_step_i = {key: val[part_i] for key, val in args_step.items()}
            args_i.update(args_step_i)
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


