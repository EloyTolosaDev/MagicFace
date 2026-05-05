import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from insightface.app import FaceAnalysis

try:
    from .data import datasets_faceswap
except ImportError:
    import data.datasets_faceswap as datasets_faceswap

try:
    from .model import BiSeNet
except ImportError:
    from model import BiSeNet


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
INSIGHTFACE_ROOT = SCRIPT_DIR / "third_party_files"
PARSING_MODEL_PATH = SCRIPT_DIR / "79999_iter.pth"
D3DFR_WEIGHTS_PATH = CHECKPOINT_DIR / "third_party" / "d3dfr_res50_nofc.pth"
BFM_MODEL_PATH = CHECKPOINT_DIR / "third_party" / "BFM_model_front.mat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = None
net = None
net_d3dfr = None
bfm_facemodel = None


RESAMPLE_BILINEAR = Image.Resampling.BILINEAR

pil2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])


def get_onnx_providers():
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


def initialize_models():
    global app, net, net_d3dfr, bfm_facemodel

    if app is not None and net is not None and net_d3dfr is not None and bfm_facemodel is not None:
        return

    if not PARSING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing parsing model at '{PARSING_MODEL_PATH}'. "
            "Download model files from the README instructions before running this script."
        )
    if not D3DFR_WEIGHTS_PATH.exists() or not BFM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing 3D face reconstruction files in 'utils/checkpoints/third_party/'. "
            "Download the model assets from the README instructions."
        )

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    try:
        import third_party.d3dfr.bfm as bfm
        import third_party.model_resnet_d3dfr as model_resnet_d3dfr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing Python modules under 'utils/third_party'. "
            "Ensure required third-party files are downloaded into the utils directory."
        ) from exc

    providers, ctx_id = get_onnx_providers()
    app = FaceAnalysis(name="antelopev2", root=str(INSIGHTFACE_ROOT), providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    net = BiSeNet(n_classes=19).to(device)
    net.load_state_dict(torch.load(PARSING_MODEL_PATH, map_location=device))
    net.eval()

    net_d3dfr = model_resnet_d3dfr.getd3dfr_res50(str(D3DFR_WEIGHTS_PATH)).eval().to(device)
    bfm_facemodel = bfm.BFM(
        focal=1015 * 256 / 224,
        image_size=256,
        bfm_model_path=str(BFM_MODEL_PATH),
    ).to(device)


def draw_pts70_batch(pts68, gaze, warp_mat256_np, dst_size, im_list=None, return_pt=False):
    draw_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    left_eye1 = pts68[:, 36]
    left_eye2 = pts68[:, 39]
    right_eye1 = pts68[:, 42]
    right_eye2 = pts68[:, 45]

    right_eye_length = torch.sqrt(torch.sum((right_eye2 - right_eye1) ** 2, dim=1, keepdim=True))
    left_eye_length = torch.sqrt(torch.sum((left_eye2 - left_eye1) ** 2, dim=1, keepdim=True))
    right_eye_center = (right_eye2 + right_eye1) * 0.5
    left_eye_center = (left_eye2 + left_eye1) * 0.5

    with torch.no_grad():
        left_gaze = gaze[:, :2] * left_eye_length + left_eye_center
        right_gaze = gaze[:, 2:] * right_eye_length + right_eye_center
        pts70 = torch.cat([pts68, left_gaze.view(-1, 1, 2), right_gaze.view(-1, 1, 2)], dim=1)
        landmarks = pts70.cpu().numpy().round().astype(int)

    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, landmarks.shape[1]))
    colors = (255 * colors).astype(int)[:, 0:3].tolist()

    im_pts70_list = []
    if im_list is None:
        im_list = [np.zeros((256, 256, 3), dtype=np.uint8) for idx in range(landmarks.shape[0])]
    else:
        im_list = [np.array(x) for x in im_list]
    for idx in range(landmarks.shape[0]):
        image = im_list[idx]

        for i in range(landmarks.shape[1]):
            if i in draw_list:
                x, y = landmarks[idx, i, :]
                color = colors[i]
                image = cv2.circle(image, (x, y), radius=2, color=(color[2], color[1], color[0]), thickness=-1)

        dst_image = cv2.warpAffine(image, warp_mat256_np[idx], (dst_size, dst_size),
                                   flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), borderMode=cv2.BORDER_CONSTANT)
        im_pts70_list.append(Image.fromarray(dst_image))

    if return_pt:
        tensor_list = [pil2tensor(x).view(1, 3, dst_size, dst_size) for x in im_pts70_list]
        batch = torch.cat(tensor_list, dim=0)
        return batch
    else:
        return im_pts70_list



def keep_background(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8) # [1, 19, 512, 512]
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi == 8  or pi == 9  or pi == 14 or pi == 17 or pi == 18 or pi == 16 or pi == 7:
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    tmp = vis_parsing_anno_color / 255
    mask_channel = 1 - tmp[:, :, 0]
    deep_gray = np.full_like(vis_im, (0, 0, 0), dtype=np.uint8)

    im_cv = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
    result_image = np.where(mask_channel[:, :, np.newaxis] == 1, deep_gray, im_cv)
    return result_image



def get_landmarks(image):
    initialize_models()
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    if len(face_info) == 0:
        return 'error'
    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]  # only use the maximum face
    pts5 = face_info['kps']

    warp_mat = datasets_faceswap.get_affine_transform(pts5, datasets_faceswap.mean_face_lm5p_256)
    drive_im_crop256 = cv2.warpAffine(np.array(image), warp_mat, (256, 256), flags=cv2.INTER_LINEAR)

    drive_im_crop256_pil = Image.fromarray(drive_im_crop256)
    image_tar_crop256 = pil2tensor(drive_im_crop256_pil).view(1, 3, 256, 256).to(device)

    gt_d3d_coeff = net_d3dfr(image_tar_crop256)
    gt_pts68, _ = bfm_facemodel.get_lm68(gt_d3d_coeff)
    image_tar_warpmat256 = warp_mat.reshape((1, 2, 3))
    im_pts70 = draw_pts70_batch(gt_pts68, gt_d3d_coeff[:, 257:], image_tar_warpmat256, 512,
                                return_pt=True)

    return im_pts70

def make_bg_for_one_image(args):
    initialize_models()
    trans = transforms.ToTensor()

    to_tensor = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(args.img_path)
        image = img.resize((512, 512), RESAMPLE_BILINEAR)
        # image = img
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]  # [1, 19, 512, 512]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        im_pts70 = get_landmarks(image)
        if isinstance(im_pts70, str):
            print('cannot find face')
            return
        else:
            im_pts70 = im_pts70[0]
            im_pts70 = torch.clamp((im_pts70 + 1) / 2, min=0, max=1)

            bg = keep_background(image, parsing, stride=1)  # cv form
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
            bg = trans(bg)  # 0-1
            res = bg + im_pts70
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(res, str(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default='',
        required=False
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='',
        required=False
    )
    args = parser.parse_args()
    make_bg_for_one_image(args)
