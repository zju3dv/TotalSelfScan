import numpy as np
import torch
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_ssim
import os
import cv2
from termcolor import colored
import lpips

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.handl_mse = []
        self.handl_psnr = []
        self.handl_ssim = []
        self.handl_lpips = []
        self.handr_mse = []
        self.handr_psnr = []
        self.handr_ssim = []
        self.handr_lpips = []
        self.face_mse = []
        self.face_psnr = []
        self.face_ssim = []
        self.face_lpips = []
        self.lpips_calculator = lpips.LPIPS(net='alex').to('cuda')

    def lpips_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        orig_img_pred = img_pred.copy()
        orig_img_gt = img_gt.copy()

        if 'crop_bbox' in batch:
            img_pred = fill_image(img_pred, batch)
            img_gt = fill_image(img_gt, batch)

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = (orig_img_pred[y:y + h, x:x + w] * 255).astype(np.uint8)
        img_gt = (orig_img_gt[y:y + h, x:x + w] * 255).astype(np.uint8)
        img_pred = lpips.im2tensor(img_pred).to('cuda')
        img_gt = lpips.im2tensor(img_gt).to('cuda')

        img_lpips = self.lpips_calculator(img_pred, img_gt).item()
        return img_lpips


    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        orig_img_pred = img_pred.copy()
        orig_img_gt = img_gt.copy()

        if 'crop_bbox' in batch:
            img_pred = fill_image(img_pred, batch)
            img_gt = fill_image(img_gt, batch)

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = orig_img_pred[y:y + h, x:x + w]
        img_gt = orig_img_gt[y:y + h, x:x + w]
        # compute the ssim

        ssim = compare_ssim(img_pred, img_gt, multichannel=True)

        return ssim

    def evaluate(self, output, batch):

        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        lpips_ = self.lpips_metric(rgb_pred, rgb_gt, batch)
        self.lpips.append(lpips_)
        if cfg.evaluation.eval_full:
            # handl
            for part in ['face', 'handl', 'handr']:
                if f'{part}_predicted_rgb' in output.keys():
                    rgb_pred = output[f'{part}_predicted_rgb'].detach().cpu().numpy()
                    rgb_gt = output[f'{part}_gt_rgb'].detach().cpu().numpy()


                    mse = np.mean((rgb_pred - rgb_gt) ** 2)
                    eval(f'self.{part}_mse').append(mse)

                    psnr = self.psnr_metric(rgb_pred, rgb_gt)
                    eval(f'self.{part}_psnr').append(psnr)

                    rgb_pred = output[f'{part}_predicted_rgb_rec'].detach().cpu().numpy()
                    rgb_gt = output[f'{part}_gt_rgb_rec'].detach().cpu().numpy()
                    # import matplotlib.pylab as plt;plt.figure();plt.imshow(rgb_pred);plt.show()

                    # import matplotlib.pylab as plt;plt.figure();plt.imshow(rgb_pred);plt.show()
                    # import matplotlib.pylab as plt;plt.figure();plt.imshow(rgb_gt);plt.show()
                    ssim = compare_ssim(rgb_pred, rgb_gt, multichannel=True)
                    eval(f'self.{part}_ssim').append(ssim)
                    rgb_pred_save = rgb_pred.copy()
                    rgb_pred = (rgb_pred * 255).astype(np.uint8)
                    rgb_gt_save = rgb_gt.copy()
                    rgb_gt = (rgb_gt * 255).astype(np.uint8)
                    rgb_pred = lpips.im2tensor(rgb_pred).to('cuda')
                    rgb_gt = lpips.im2tensor(rgb_gt).to('cuda')
                    lpips_ = self.lpips_calculator(rgb_pred, rgb_gt).item()
                    eval(f'self.{part}_lpips').append(lpips_)

                result_dir = os.path.join(cfg.result_dir, 'comparison_{}'.format(part))
                os.system('mkdir -p {}'.format(result_dir))
                frame_index = batch['frame_index'].item()
                view_index = batch['cam_ind'].item()
                cv2.imwrite(
                    '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                           view_index),
                    (rgb_pred_save[..., [2, 1, 0]] * 255))
                cv2.imwrite(
                    '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                              view_index),
                    (rgb_gt_save[..., [2, 1, 0]] * 255))

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        if cfg.evaluation.eval_full:
            metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips,
                       'handl_mse': self.handl_mse, 'handl_psnr': self.handl_psnr, 'handl_ssim': self.handl_ssim, 'handl_lpips': self.handl_lpips,
                       'handr_mse': self.handr_mse, 'handr_psnr': self.handr_psnr, 'handr_ssim': self.handr_ssim, 'handr_lpips': self.handr_lpips,
                       'face_mse': self.face_mse, 'face_psnr': self.face_psnr, 'face_ssim': self.face_ssim, 'face_lpips': self.face_lpips,}
            np.save(result_path, metrics)
            print('mse: {}'.format(np.mean(self.mse)))
            print('psnr: {}'.format(np.mean(self.psnr)))
            print('ssim: {}'.format(np.mean(self.ssim)))
            print('lpips: {}'.format(np.mean(self.lpips)))
            print('face_mse: {}'.format(np.mean(self.face_mse)))
            print('face_psnr: {}'.format(np.mean(self.face_psnr)))
            print('face_ssim: {}'.format(np.mean(self.face_ssim)))
            print('face_lpips: {}'.format(np.mean(self.face_lpips)))
            print('handl_mse: {}'.format(np.mean(self.handl_mse)))
            print('handl_psnr: {}'.format(np.mean(self.handl_psnr)))
            print('handl_ssim: {}'.format(np.mean(self.handl_ssim)))
            print('handl_lpips: {}'.format(np.mean(self.handl_lpips)))
            print('handr_mse: {}'.format(np.mean(self.handr_mse)))
            print('handr_psnr: {}'.format(np.mean(self.handr_psnr)))
            print('handr_ssim: {}'.format(np.mean(self.handr_ssim)))
            print('handr_lpips: {}'.format(np.mean(self.handr_lpips)))
            self.mse = []
            self.psnr = []
            self.ssim = []
        else:
            metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
            np.save(result_path, metrics)
            print('mse: {}'.format(np.mean(self.mse)))
            print('psnr: {}'.format(np.mean(self.psnr)))
            print('ssim: {}'.format(np.mean(self.ssim)))
            self.mse = []
            self.psnr = []
            self.ssim = []


def fill_image(img, batch):
    orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
    full_img = np.zeros((orig_H, orig_W, 3))
    bbox = batch['crop_bbox'][0].detach().cpu().numpy()
    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    full_img[bbox[0, 1]:bbox[1, 1],
             bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
    return full_img
