import os

from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'Xuzhen_Train': {
            'data_root': 'data/zju_snapshot/male-xz-smplh',
            'human': 'male-xz-smplh',
            'ann_file': 'data/zju_snapshot/male-xz-smplh/annots.npy',
            'split': 'train'
        },
        'Xuzhen_Test': {
            'data_root': 'data/zju_snapshot/male-xz-smplh',
            'human': 'male-xz-smplh',
            'ann_file': 'data/zju_snapshot/male-xz-smplh/annots.npy',
            'split': 'test'
        },
        'Shuaiqing_Train': {
            'data_root': 'data/zju_snapshot/male-sq-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-sq-smplh/annots.npy',
            'split': 'train'
        },
        'Shuaiqing_Test': {
            'data_root': 'data/zju_snapshot/male-sq-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-sq-smplh/annots.npy',
            'split': 'test'
        },
        'Ting_Train': {
            'data_root': 'data/zju_snapshot/male-ting-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-ting-smplh/annots.npy',
            'split': 'train'
        },
        'Ting_Test': {
            'data_root': 'data/zju_snapshot/male-ting-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-ting-smplh/annots.npy',
            'split': 'test'
        },
        'Ting315_Train': {
            'data_root': 'data/zju_snapshot/male-ting315-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-ting315-smplh/annots.npy',
            'split': 'train'
        },
        'Ting315_Test': {
            'data_root': 'data/zju_snapshot/male-ting315-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-ting315-smplh/annots.npy',
            'split': 'test'
        },
        'Djt330_Train': {
            'data_root': 'data/zju_snapshot/male-djt330-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt330-smplh/annots.npy',
            'split': 'train'
        },
        'Djt330_Test': {
            'data_root': 'data/zju_snapshot/male-djt330-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt330-smplh/annots.npy',
            'split': 'test'
        },
        'Djt330_2_Train': {
            'data_root': 'data/zju_snapshot/male-djt330_2-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt330_2-smplh/annots.npy',
            'split': 'train'
        },
        'Djt330_2_Test': {
            'data_root': 'data/zju_snapshot/male-djt330_2-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt330_2-smplh/annots.npy',
            'split': 'test'
        },
        'Djtgo410_Train': {
            'data_root': 'data/zju_snapshot/male-djtgo410-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djtgo410-smplh/annots.npy',
            'split': 'train'
        },
        'Djtgo410_Test': {
            'data_root': 'data/zju_snapshot/male-djtgo410-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djtgo410-smplh/annots.npy',
            'split': 'test'
        },
        'Djt_shirt_Train': {
            'data_root': 'data/zju_snapshot/male-djt_shirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt_shirt-smplh/annots.npy',
            'split': 'train'
        },
        'Djt_shirt_Test': {
            'data_root': 'data/zju_snapshot/male-djt_shirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt_shirt-smplh/annots.npy',
            'split': 'test'
        },
        'Djt_newshirt_Train': {
            'data_root': 'data/zju_snapshot/male-djt_newshirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt_newshirt-smplh/annots.npy',
            'split': 'train'
        },
        'Djt_newshirt_Test': {
            'data_root': 'data/zju_snapshot/male-djt_newshirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt_newshirt-smplh/annots.npy',
            'split': 'test'
        },
        'Djt415_shirt_Train': {
            'data_root': 'data/zju_snapshot/male-djt415_shirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_shirt-smplh/annots.npy',
            'split': 'train'
        },
        'Djt415_shirt_Test': {
            'data_root': 'data/zju_snapshot/male-djt415_shirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_shirt-smplh/annots.npy',
            'split': 'test'
        },
        'Djt415_tshirt_Train': {
            'data_root': 'data/zju_snapshot/male-djt415_tshirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_tshirt-smplh/annots.npy',
            'split': 'train'
        },
        'Djt415_tshirt_Test': {
            'data_root': 'data/zju_snapshot/male-djt415_tshirt-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_tshirt-smplh/annots.npy',
            'split': 'test'
        },
        'Djt415_m5_Train': {
            'data_root': 'data/zju_snapshot/male-djt415_m5-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m5-smplh/annots.npy',
            'split': 'train'
        },
        'Djt415_m5_Test': {
            'data_root': 'data/zju_snapshot/male-djt415_m5-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m5-smplh/annots.npy',
            'split': 'test'
        },
        'Djt415_m4_Train': {
            'data_root': 'data/zju_snapshot/male-djt415_m4-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m4-smplh/annots.npy',
            'split': 'train'
        },
        'Djt415_m4_Test': {
            'data_root': 'data/zju_snapshot/male-djt415_m4-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m4-smplh/annots.npy',
            'split': 'test'
        },
        'Djt415_m3_Train': {
            'data_root': 'data/zju_snapshot/male-djt415_m3-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m3-smplh/annots.npy',
            'split': 'train'
        },
        'Djt415_m3_Test': {
            'data_root': 'data/zju_snapshot/male-djt415_m3-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-djt415_m3-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi1_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi1-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi1-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi1_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi1-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi1-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi3_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi3-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi3-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi3_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi3-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi3-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi4_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi4-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi4-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi4_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi4-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi4-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi7_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi7-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi7-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi7_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi7-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi7-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi8_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi8-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi8-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi8_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi8-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi8-smplh/annots.npy',
            'split': 'test'
        },
        'Syn_mi9_Train': {
            'data_root': 'data/zju_snapshot/male-syn_mi9-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi9-smplh/annots.npy',
            'split': 'train'
        },
        'Syn_mi9_Test': {
            'data_root': 'data/zju_snapshot/male-syn_mi9-smplh',
            'human': 'male-tzr-smplh',
            'ann_file': 'data/zju_snapshot/male-syn_mi9-smplh/annots.npy',
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
