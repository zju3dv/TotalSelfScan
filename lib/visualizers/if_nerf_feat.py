import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import open3d as o3d
from lib.utils import vis_utils


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


class Visualizer:
    def __init__(self):
        self.pts = []
        self.feat = []
        self.mesh = []
        self.ni = 0

    def visualize(self, output, batch):
        self.pts.append(output['pts'])
        self.feat.append(output['feat'])
        faces = np.load('data/light_stage/smpl_faces.npy')
        self.mesh.append(get_o3d_mesh(output['pts'], faces))
        self.ni = self.ni + 1

        pc = o3d.geometry.PointCloud()
        pts = np.load('/home/pengsida/Datasets/light_stage/vertices/CoreView_315/{}.npy'.format(self.ni))
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(np.load('color.npy'))
        # o3d.visualization.draw_geometries([pc])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()
        vis.add_geometry(pc)

        o3d_param = np.load('o3d_param.npy', allow_pickle=True).item()
        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = o3d_param['extri']
        param.intrinsic.set_intrinsics(*o3d_param['intri'])
        ctr.convert_from_pinhole_camera_parameters(param)

        # vis.run()
        # param = ctr.convert_to_pinhole_camera_parameters()
        # extrinsic = param.extrinsic
        # intrinsic = param.intrinsic
        # width, height = intrinsic.width, intrinsic.height
        # fx, fy = intrinsic.get_focal_length()
        # cx, cy = intrinsic.get_principal_point()
        # np.save('o3d_param.npy', {'extri': extrinsic, 'intri': [width, height, fx, fy, cx, cy]})

        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image('/home/pengsida/Downloads/point_cloud/{}.jpg'.format(self.ni))
        vis.destroy_window()

        # if self.ni == 1:
        #     feat = np.concatenate(self.feat)
        #     colors = vis_utils.tsne_colors(feat)
        #     colors = np.split(colors, self.ni)

        #     pcs = []
        #     for i in range(len(self.pts)):
        #         pts_ = self.pts[i].copy()
        #         pts_[:, 0] += i * 1.5
        #         colors_ = colors[i]
        #         pc = o3d.geometry.PointCloud()
        #         pc.points = o3d.utility.Vector3dVector(pts_)
        #         pc.colors = o3d.utility.Vector3dVector(colors_)
        #         pcs.append(pc)

        #     o3d.visualization.draw_geometries(pcs)
