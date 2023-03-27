import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from lib.utils.base_utils import *
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import pytorch3d.structures as struct


class Renderer:
    def __init__(self, net):
        smplh = read_pickle('data/smplx/smplh/SMPLH_MALE.pkl')
        self.bw = torch.from_numpy(smplh['weights'].T)[None].cuda().float()
        self.tface = torch.from_numpy(smplh['f'].astype(np.float32)).cuda()
        self.net = net

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

        wpts = wpts.view(1, n_pixel * n_sample, 3)
        num_joints = wpts_bw.shape[-1]
        wpts_bw = wpts_bw.view(1, n_pixel * n_sample, num_joints).permute(0, 2, 1)
        wpts_ray_d = wpts_ray_d.view(1, n_pixel * n_sample, 3)

        # transform points from the world space to the pose space
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        pose_dirs = world_dirs_to_pose_dirs(wpts_ray_d, batch['R'])

        # transform points from the pose space to the tpose space
        tpose = pose_points_to_tpose_points(pose_pts, wpts_bw, batch['A'])
        bigpose = tpose_points_to_pose_points(tpose, wpts_bw, batch['big_A'])
        init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, wpts_bw, batch['A'])
        tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, wpts_bw,
                                             batch['big_A'])

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

    def render(self, batch):
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
        # import ipdb; ipdb.set_trace(context=11)
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
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device='cuda',
                    cameras=cameras)
            )

            # pose the mesh
            ppose = pose_points_to_tpose_points(batch['tpose'], self.bw,
                                                batch['big_A'])
            # from lib.utils.debugger import dbg
            # dbg.showL3D([ppose[0].cpu(), batch['tpose'][0].cpu()])
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
            mask = face_idx_map > 0
            verts_rgb = torch.full([vertex.shape[0], vertex.shape[1], 3], 0.5, device='cuda')
            ppose.textures = TexturesVertex(verts_features=verts_rgb)
            img = renderer(ppose)[0, ..., :3]
            input_img = batch['img'][0].permute([1, 2, 0])
            input_img[mask] = img[mask]
            # import matplotlib.pylab as plt;plt.figure();plt.imshow(input_img.cpu());plt.show()

        ret = {}
        rgb_map = input_img[None]
        ret['rgb_map'] = rgb_map

        return ret
