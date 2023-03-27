import os
import torch
import torch.nn.functional as F
from lib.config import cfg
from tqdm import tqdm
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
import pytorch3d.structures as struct
from lib.utils.total_utils import get_inside


class Renderer:
    def __init__(self, net):
        tpose_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.exp_name)
        if cfg.latent_optim:
            os.makedirs(os.path.dirname(tpose_dir), exist_ok=True)
            init_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.init_sdf)
            cmd = 'cp {} {}'.format(init_dir, tpose_dir)
            os.system(cmd)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        import trimesh
        mesh = trimesh.Trimesh(vertices=tpose_mesh['vertex'], faces=tpose_mesh['triangle'], process=False)
        self.net = net
        self.tvertex = torch.from_numpy(tpose_mesh['vertex']).cuda().float()
        triangle = tpose_mesh['triangle'].astype(np.int32)
        self.tface = torch.from_numpy(triangle).cuda().float()
        self.bw = torch.from_numpy(
            tpose_mesh['blend_weight'][:-1]).cuda()[None].float()
        self.mesh_edges = mesh.edges_unique
        self.mesh_edges_length = mesh.edges_unique_length
        self.augment_view_dirs = True
        self.init = True
        self.selected_tpose_dirs = {}

    def get_wsampling_points(self, ray_o, ray_d, wpts):
        # calculate the steps for each ray
        n_batch, num_rays = ray_d.shape[:2]
        z_interval = cfg.z_interval
        n_sample = cfg.N_samples // 2 * 2 + 1
        # 这里sample的是前后z_interval距离的点，如果z_interval太大，就会采到其他的点
        z_vals = torch.linspace(-z_interval, z_interval,
                                steps=n_sample).to(ray_d)

        if cfg.perturb > 0. and self.net.training:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(ray_d)
            z_vals = lower + (upper - lower) * t_rand

        pts = wpts[:, :, None] + ray_d[:, :, None] * z_vals[:, None]
        z_vals = (pts[..., 0] - ray_o[..., :1]) / ray_d[..., :1]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        ret = raw_decoder(wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch, face_idx,
                        bary_coords):
        n_batch = ray_o.shape[0]
        n_pixel = face_idx.shape[1]
        # sampling points for nerf training
        with torch.no_grad():
            pixel_vertex_idx = self.tface[face_idx].long()

            # get 3d points
            pixel_vertex = self.pvertex[pixel_vertex_idx, :]
            wpts = torch.sum(pixel_vertex * bary_coords[..., None], dim=2)

            # get blend weights of 3d points
            pixel_bw = self.bw[0].transpose(0, 1)[pixel_vertex_idx]
            wpts_bw = torch.sum(pixel_bw * bary_coords[..., None], dim=2)

            wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, wpts)
            n_sample = wpts.shape[2]
            wpts_bw = wpts_bw[:, :, None].expand(-1, -1, n_sample,
                                                 -1).contiguous()
            wpts_ray_d = ray_d[:, :, None].expand(-1, -1, n_sample,
                                                  -1).contiguous()
            full_face_idx = face_idx[:, :, None].expand(-1, -1, n_sample).contiguous()
            full_bary_coords = bary_coords[:, :, None].expand(-1, -1, n_sample, -1).contiguous()


        wpts = wpts.view(1, n_pixel * n_sample, 3)
        num_joints = wpts_bw.shape[-1]
        wpts_bw = wpts_bw.view(1, n_pixel * n_sample, num_joints).permute(0, 2, 1)
        wpts_ray_d = wpts_ray_d.view(1, n_pixel * n_sample, 3)
        full_face_idx = full_face_idx.view(1, n_pixel * n_sample)
        full_bary_coords = full_bary_coords.view(1, n_pixel * n_sample, 3)

        # transform points from the world space to the pose space
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        pose_dirs = world_dirs_to_pose_dirs(wpts_ray_d, batch['R'])

        # transform points from the pose space to the tpose space
        tpose = pose_points_to_tpose_points(pose_pts, wpts_bw, batch['A'])
        bigpose = tpose_points_to_pose_points(tpose, wpts_bw, batch['big_A'])
        # all_view_dirs = []
        # for i in self.view_dirs_statics[0].keys():
        #     all_view_dirs.append(self.view_dirs_statics[0][i])
        # all_view_dirs = torch.vstack(all_view_dirs)
        init_tdirs = pose_dirs_to_tpose_dirs_rigid(pose_dirs, wpts_bw, batch['A'])
        tpose_dirs = tpose_dirs_to_pose_dirs_rigid(init_tdirs, wpts_bw,
                                             batch['big_A'])
        fig_index = batch['latent_index'].item()
        if cfg.select_view and batch['part_type'].item()!=0: #and not (fig_index in self.selected_tpose_dirs.keys()):
            from lib.utils.total_utils_base import judge_bounds
            wpts_part_label = judge_bounds(bigpose[0], batch)[None]
            self.transform_view_dirs_vertex_batch(tpose_dirs, full_face_idx, wpts_part_label, full_bary_coords)
            # self.selected_tpose_dirs[fig_index] = tpose_dirs

        # if cfg.select_view and batch['part_type'].item()!=0 and fig_index in self.selected_tpose_dirs.keys():
        #     tpose_dirs = self.selected_tpose_dirs[fig_index]

        # compute the color and density
        ret = self.net.tpose_human(bigpose[0], tpose_dirs[0], None, batch)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        n_channel = ret['raw'].size(1)
        raw = ret['raw'].reshape(-1, n_sample, n_channel)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret = ({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
        })

        return ret

    def get_face_idx(self, batch):
        part_type = cfg.part_type
        part_names = ['body', 'face', 'handl', 'handr']
        bound_name = 'tbounds_{}'.format(part_names[part_type])
        # obtain the verts id in each part bbox
        idx_in_part = get_inside(self.tvertex, batch[bound_name][0])
        idx_in_body = get_inside(self.tvertex, batch['tbounds_body'][0])
        idx_in_overlap = idx_in_part * idx_in_body
        face_idx_in_part = idx_in_overlap[self.tface.long()]
        face_idx_in_part = face_idx_in_part.sum(dim=1) == 3
        
        return face_idx_in_part

    def transform_view_dirs_vertex_batch(self, tpose_dirs, face_idx, face_part_label, bary_coords):
        # face_part_label = face_part_label[face_idx]
        part_type = [0, 1, 2, 3]
        for part_i in part_type:
            index = (face_part_label[0] == part_i)
            if index.sum().item() == 0:
                continue
            vertex_batch = self.tface[face_idx[0][index]]
            bary_coords_batch = bary_coords[0, index]
            mean_view_dirs_batch = self.view_dirs_statics[part_i][vertex_batch.to(torch.long)]
            weighted_view_dirs_batch = torch.einsum('njk,nj->nk', mean_view_dirs_batch, bary_coords_batch)
            weighted_view_dirs_batch /= weighted_view_dirs_batch.norm(dim=1, keepdim=True)
            tpose_dirs[0, index] = weighted_view_dirs_batch

    def transform_view_dirs_vertex(self, tpose_dirs, face_idx, face_part_label, bary_coords):
        # face_part_label = face_part_label[face_idx]
        for idx in range(tpose_dirs.shape[1]):
            part_label = int(face_part_label[0, idx].item())
            face = face_idx[0, idx].item()
            current_view_dir = tpose_dirs[0, idx]
            vertex = self.tface[face]
            current_bary_coords = bary_coords[0, idx]
            select_view_dir = {}
            for vertex_idx, vertex_i in enumerate(vertex):
                vertex_i = vertex_i.item()
                # face_adjacent, _ = torch.where(self.tface==vertex_i)
                # vert_adjacent = torch.unique(self.tface[face_adjacent])
                # face_adjacent_double = torch.cat([torch.where(self.tface==i.item())[0] for i in vert_adjacent])
                # vert_adjacent_double = torch.unique(self.tface[face_adjacent_double])
                if vertex_i in self.view_dirs_statics[part_label].keys():
                    # existing_view_dirs = torch.cat([self.view_dirs_statics[part_label][i.item()] for i in vert_adjacent_double], dim=0)
                    if cfg.debug.use_mean_view_dir:
                        existing_view_dirs = self.view_dirs_statics[part_label][vertex_i]
                        existing_view_dirs_normalized = existing_view_dirs / existing_view_dirs.norm(dim=-1,
                                                                                                     keepdim=True)
                        mean_view_dir = existing_view_dirs_normalized.mean(dim=0)
                        select_view_dir[vertex_idx] = mean_view_dir/mean_view_dir.norm()

                    else:
                        existing_view_dirs = self.view_dirs_statics[part_label][vertex_i]
                        existing_view_dirs_normalized = existing_view_dirs / existing_view_dirs.norm(dim=-1, keepdim=True)
                        similarity_normalized = existing_view_dirs_normalized @ current_view_dir
                        max_position_normalized = similarity_normalized.argmax().item()
                        select_view_dir[vertex_idx] = existing_view_dirs[max_position_normalized]

                    # if similarity[max_position] > 1:
                    #     continue
                    # current_view_dir[:] = existing_view_dirs[max_position]
            if len(select_view_dir.keys()) > 0:
                sum_view_dir = 0
                sum_weight = 0
                for i in select_view_dir.keys():
                    sum_view_dir = sum_view_dir + select_view_dir[i] * current_bary_coords[i]
                    sum_weight = sum_weight + current_bary_coords[i]
                target_view_dir = sum_view_dir / sum_weight
                target_view_dir /= target_view_dir.norm()
                current_view_dir[:] = target_view_dir

    def augment_view_dirs_statics_vertex(self, view_dirs_path, view_dirs_statics, edges, edges_length, batch, part):
        part_name = ['body', 'face', 'handl', 'handr']
        num_vertex = edges.max().item() + 1

        view_dirs_body_path = view_dirs_path.replace('view_dirs.npy', f'view_dirs_vertex_{part_name[part]}.npy')
        if os.path.exists(view_dirs_body_path):
            view_dirs_statics[part] = np.load(view_dirs_body_path, allow_pickle=True).item()
        else:
            num_existing_vertex = len(view_dirs_statics[part].keys())
            while(num_existing_vertex<num_vertex):
                print('process {}/{}'.format(num_existing_vertex, num_vertex))
                existing_vertex = np.array(list(view_dirs_statics[part].keys()))
                vertex_isin = np.isin(np.arange(num_vertex), existing_vertex)
                edges_isin = vertex_isin[edges]
                cross_edges_idx = np.where(edges_isin.sum(axis=1)==1)[0]
                cross_edges = edges[cross_edges_idx]
                cross_edges_length = edges_length[cross_edges_idx]
                adjacent_vertex = cross_edges[~edges_isin[cross_edges_idx]]
                adjacent_existing_vertex = cross_edges[edges_isin[cross_edges_idx]]
                adjacent_vertex_unique = np.unique(adjacent_vertex)
                for vertex in adjacent_vertex_unique:
                    select_vertex_idx = np.where(adjacent_vertex == vertex)[0]
                    select_edges_length = cross_edges_length[select_vertex_idx]
                    select_one = np.argmin(select_edges_length)
                    select_vertex = adjacent_existing_vertex[select_vertex_idx[select_one]]
                    view_dirs_statics[part][vertex] = view_dirs_statics[part][select_vertex]
                num_existing_vertex = len(view_dirs_statics[part].keys())

            for vertex_idx in tqdm(range(num_vertex)):
                if batch['vertex_label'][0, vertex_idx] == part and vertex_idx not in view_dirs_statics[part].keys():
                    view_dirs_statics[part][vertex_idx] = self.get_closest_adjacent_vertex_view_dirs(vertex_idx, edges, edges_length, view_dirs_statics[part])
            np.save(view_dirs_body_path, view_dirs_statics[part])

    def render(self, batch):
        if cfg.select_view and self.init:
            if cfg.view_dir_path != '':
                view_dirs_path = cfg.view_dir_path
            else:
                view_dirs_path = 'data/view_dirs/{}/view_dirs.npy'.format(cfg.exp_name)
            view_dirs_statics = np.load(view_dirs_path, allow_pickle=True).item()
            if self.augment_view_dirs:
                if False:
                    self.augment_view_dirs_statics(view_dirs_path, view_dirs_statics, self.num_face, self.face_adjacency_np, batch, part=0)
                    self.augment_view_dirs_statics(view_dirs_path, view_dirs_statics, self.num_face, self.face_adjacency_np, batch, part=2)
                    self.augment_view_dirs_statics(view_dirs_path, view_dirs_statics, self.num_face, self.face_adjacency_np, batch, part=3)
                else:
                    self.augment_view_dirs_statics_vertex(view_dirs_path, view_dirs_statics, self.mesh_edges, self.mesh_edges_length, batch, part=0)
                    self.augment_view_dirs_statics_vertex(view_dirs_path, view_dirs_statics, self.mesh_edges, self.mesh_edges_length, batch, part=1)
                    self.augment_view_dirs_statics_vertex(view_dirs_path, view_dirs_statics, self.mesh_edges, self.mesh_edges_length, batch, part=2)
                    self.augment_view_dirs_statics_vertex(view_dirs_path, view_dirs_statics, self.mesh_edges, self.mesh_edges_length, batch, part=3)

            body_view = view_dirs_statics[0]
            for face_idx in body_view.keys():
                body_view[face_idx] = torch.from_numpy(body_view[face_idx]).cuda().float()
                if cfg.debug.use_mean_view_dir:
                    body_view[face_idx] = body_view[face_idx].mean(dim=0, keepdim=True)
            if cfg.debug.use_mean_view_dir:
                body_view_batch = []
                for i in range(int(max(list(body_view.keys())))+1):
                    body_view_batch.append(body_view[i][0])
                body_view = torch.stack(body_view_batch)
            face_view = view_dirs_statics[1]
            for face_idx in face_view.keys():
                face_view[face_idx] = torch.from_numpy(face_view[face_idx]).cuda().float()
                if cfg.debug.use_mean_view_dir:
                    face_view[face_idx] = face_view[face_idx].mean(dim=0, keepdim=True)
            if cfg.debug.use_mean_view_dir:
                face_view_batch = []
                for i in range(int(max(list(face_view.keys())))+1):
                    face_view_batch.append(face_view[i][0])
                face_view = torch.stack(face_view_batch)
            handl_view = view_dirs_statics[2]
            for face_idx in handl_view.keys():
                handl_view[face_idx] = torch.from_numpy(handl_view[face_idx]).cuda().float()
                if cfg.debug.use_mean_view_dir:
                    handl_view[face_idx] = handl_view[face_idx].mean(dim=0, keepdim=True)
            if cfg.debug.use_mean_view_dir:
                handl_view_batch = []
                for i in range(int(max(list(handl_view.keys())))+1):
                    handl_view_batch.append(handl_view[i][0])
                handl_view = torch.stack(handl_view_batch)
            handr_view = view_dirs_statics[3]
            for face_idx in handr_view.keys():
                handr_view[face_idx] = torch.from_numpy(handr_view[face_idx]).cuda().float()
                if cfg.debug.use_mean_view_dir:
                    handr_view[face_idx] = handr_view[face_idx].mean(dim=0, keepdim=True)
            if cfg.debug.use_mean_view_dir:
                handr_view_batch = []
                for i in range(int(max(list(handr_view.keys())))+1):
                    handr_view_batch.append(handr_view[i][0])
                handr_view = torch.stack(handr_view_batch)
            self.view_dirs_statics = {0: body_view,
                                      1: face_view,
                                      2: handl_view,
                                      3: handr_view}
            self.init = False
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # volume rendering for each pixel
        n_batch = ray_o.shape[0]
        ret_list = []
        pytorch3d_K = batch['pytorch3d_K']
        R = batch['cam_R']
        T = batch['cam_T']
        can_idx = batch['cam_ind']
        # obtain the verts id in each part bbox
        face_idx_in_overlap = self.get_face_idx(batch)

        # choose the handl pixel in the image for latent code optimization
        with torch.no_grad():
            # set camera
            cameras = PerspectiveCameras(device=ray_o.device,
                                         K=pytorch3d_K.float(),
                                         R=R[0].T[None],
                                         T=T[0].T.float())
            height, width = batch['img'].shape[-2:]
            raster_settings = RasterizationSettings(image_size=(height, width),
                                                    blur_radius=0.0,
                                                    faces_per_pixel=1,
                                                    bin_size=None)
            rasterizer = MeshRasterizer(cameras=cameras,
                                        raster_settings=raster_settings)

            # pose the mesh
            ppose = pose_points_to_tpose_points(self.tvertex[None], self.bw,
                                                batch['big_A'])
            if hasattr(self.net, 'getOptimizedTransformationMatrixAndDeltaPose'):
                A, _ = self.net.getOptimizedTransformationMatrixAndDeltaPose(batch)
                A = A[None]
            else:
                A = batch['A']
            pvertex_i = tpose_points_to_pose_points(ppose, self.bw, A)
            vertex = pose_points_to_world_points(pvertex_i, batch['R'],
                                                 batch['Th'])
            self.pvertex = vertex[0]

            # perform the rasterization
            ppose = struct.Meshes(verts=vertex, faces=self.tface[None])
            fragments = rasterizer(ppose)
            face_idx_map = fragments.pix_to_face[0, ..., 0]
            part_mask = face_idx_in_overlap[face_idx_map]
            mask = face_idx_map > 0
            mask = mask * part_mask
            # import ipdb; ipdb.set_trace(context=11)
            # import matplotlib.pylab as plt;plt.figure();plt.imshow(mask.cpu());plt.show()
            num_samples = mask.sum()
            upper_bound = 2048*10
            if num_samples > upper_bound:
                rand_samples = torch.randperm(num_samples)[:(num_samples-upper_bound)].to(face_idx_map)
                mask_ = mask.reshape(-1)
                position = torch.where(mask_==True)[0]
                mask_[position[rand_samples]] = False
                mask= mask_.reshape(mask.shape)
            face_idx_map = face_idx_map[mask][None]
            bary_coords_map = fragments.bary_coords[0, :, :, 0]
            bary_coords_map = bary_coords_map[mask][None]

            ray_o = ray_o[0][mask][None]
            ray_d = ray_d[0][mask][None]
            n_pixel = ray_o.shape[1]

        chunk = 2048 * 3
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            face_idx_chunk = face_idx_map[:, i:i + chunk]
            bary_coords_chunk = bary_coords_map[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch,
                                               face_idx_chunk,
                                               bary_coords_chunk)
            ret_list.append(pixel_value)
        if cfg.latent_optim:
            with torch.no_grad():
                batch['part_type'] = torch.zeros_like(batch['part_type'])
                ret_body_list = []
                for i in range(0, n_pixel, chunk):
                    ray_o_chunk = ray_o[:, i:i + chunk]
                    ray_d_chunk = ray_d[:, i:i + chunk]
                    near_chunk = near[:, i:i + chunk]
                    far_chunk = far[:, i:i + chunk]
                    face_idx_chunk = face_idx_map[:, i:i + chunk]
                    bary_coords_chunk = bary_coords_map[:, i:i + chunk]
                    pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                                       near_chunk, far_chunk, batch,
                                                       face_idx_chunk,
                                                       bary_coords_chunk)
                    ret_body_list.append(pixel_value)
        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
        ret['render_mask'] = mask[None]
        mask_at_box = batch['mask_at_box']
        H, W = mask_at_box.shape[1:]
        pixel_rgb_map = ret['rgb_map']
        ret['rgb_gt'] = batch['img'].permute(0, 2, 3, 1)[mask[None]][None]
        ret['rgb_map'] = pixel_rgb_map
        if cfg.latent_optim:
            ret_body = {k: torch.cat([r[k] for r in ret_body_list], dim=1) for k in keys}
            pixel_rgb_map = ret_body['rgb_map']
            ret['rgb_gt'] = pixel_rgb_map

        return ret
