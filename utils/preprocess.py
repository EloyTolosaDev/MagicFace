import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "magicface_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

try:
    from .face_analysis import get_face_analysis_app
    from .data import datasets_faceswap
except ImportError:
    from face_analysis import get_face_analysis_app
    import data.datasets_faceswap as datasets_faceswap


pil2tensor = transforms.ToTensor()


def get_bbox(dets, crop_ratio):
    if crop_ratio > 0:
        bbox = dets[0:4]
        bbox_size = max(bbox[2] - bbox[0], bbox[2] - bbox[0])
        bbox_x = 0.5 * (bbox[2] + bbox[0])
        bbox_y = 0.5 * (bbox[3] + bbox[1])
        x1 = bbox_x - bbox_size * crop_ratio
        x2 = bbox_x + bbox_size * crop_ratio
        y1 = bbox_y - bbox_size * crop_ratio
        y2 = bbox_y + bbox_size * crop_ratio
        bbox_pts4 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
    else:
        # original box
        bbox = dets[0:4].reshape((2, 2))
        bbox_pts4 = datasets_faceswap.get_box_lm4p(bbox)
    return bbox_pts4



def crop_one_image(args):
    app = get_face_analysis_app()
    cur_img_sor_path = args.img_path
    im_pil_sor = Image.open(cur_img_sor_path).convert("RGB")
    face_info_sor = app.get(cv2.cvtColor(np.array(im_pil_sor), cv2.COLOR_RGB2BGR))
    assert len(face_info_sor) >= 1, 'The input image must contain a face！'
    if len(face_info_sor) > 1:
        print('The input image contain more than one face, we will only use the maximum face')
    face_info_sor = sorted(
        face_info_sor,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]
    dets_sor= face_info_sor['bbox']
    bbox_pst_sor = get_bbox(dets_sor, crop_ratio=0.75)

    warp_mat_crop_sor = datasets_faceswap.transformation_from_points(bbox_pst_sor,
                                                                     datasets_faceswap.mean_box_lm4p_512)
    im_crop512_sor = cv2.warpAffine(np.array(im_pil_sor), warp_mat_crop_sor, (512, 512), flags=cv2.INTER_LINEAR)

    im_pil_sor = Image.fromarray(im_crop512_sor)
    im_pil_sor = pil2tensor(im_pil_sor)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(im_pil_sor, str(save_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    crop_one_image(args)
