# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend
import os
import pdb

if __name__ == '__main__':
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("./models/segm_models/pointrend_rcnn_R_101_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "./models/segm_models/pointrend_rcnn_R_101_FPN_3x_coco.pkl"
    predictor = DefaultPredictor(cfg)

    seq_names = ['s1', 's2', 's3', 's4']
    for seq_name in seq_names:
        img_path = './dataset/'+seq_name+'/stream/'
        npy_name = './dataset/'+seq_name+'/masks.npy'
        img_files = sorted(os.listdir(img_path))
        avail_path = './dataset/'+seq_name+'/avail_frms.npy'
        avail_frms = np.load(avail_path)
        mask_seq = {}
        for img_file in img_files:
            frame_num = int(img_file[:5])-1
            if frame_num not in avail_frms:
                continue
            print(img_file)
            im = cv2.imread(os.path.join(img_path, img_file))

            outputs = predictor(im)
            pred_classes = outputs['instances'].pred_classes.tolist()
            try:
                person_index = pred_classes.index(0)
                pred_mask = outputs['instances'].pred_masks[person_index].cpu().numpy()
                human_inds = np.nonzero(pred_mask)
                save_inds = np.stack((human_inds[0], human_inds[1])).transpose()
                mask_seq[str(frame_num)] = save_inds.astype(np.int16)
            except ValueError:
                print('ValueError, saving None...')
                mask_seq[str(frame_num)] = None

        np.save(npy_name, mask_seq)
