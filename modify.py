import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run the full MagicFace image modification pipeline.")
    parser.add_argument("--img_path", type=Path, required=True, help="Path to the original face image.")
    parser.add_argument("--au_test", type=str, required=True, help="AU names to edit, separated by '+'.")
    parser.add_argument("--AU_variation", type=str, required=True, help="AU intensities, separated by '+'.")
    parser.add_argument("--saved_path", type=Path, default=Path("edited_images"), help="Directory for generated images.")
    parser.add_argument("--no_crop", action="store_true", help="Skip face crop preprocessing and use --img_path directly.")
    parser.add_argument(
        "--work_dir",
        "--workdir",
        dest="work_dir",
        type=Path,
        default=None,
        help="Optional directory to keep intermediate crop/background files. Final images use --saved_path.",
    )
    parser.add_argument("--seed", type=int, default=424, help="Seed for reproducible inference.")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of diffusion inference steps.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="Stable Diffusion model path or Hugging Face model id.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--denoising_unet_path", type=str, default="mengtingwei/magicface")
    parser.add_argument("--ID_unet_path", type=str, default="mengtingwei/magicface")
    parser.add_argument(
        "--require_onnx_cuda",
        action="store_true",
        help="Fail if InsightFace ONNX models cannot use CUDAExecutionProvider.",
    )
    return parser.parse_args(input_args)


def build_intermediate_paths(img_path, work_dir):
    source_path = Path(img_path)
    suffix = source_path.suffix or ".png"
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    return work_path / f"{source_path.stem}_crop{suffix}", work_path / f"{source_path.stem}_bg.png"


def run_pipeline(args, work_dir):
    if args.require_onnx_cuda:
        os.environ["MAGICFACE_REQUIRE_ONNX_CUDA"] = "1"

    from inference import main as run_inference
    from inference import parse_args as parse_inference_args
    from utils.retrieve_bg import make_bg_for_one_image

    crop_path, bg_path = build_intermediate_paths(args.img_path, work_dir)

    inference_img_path = args.img_path
    if not args.no_crop:
        from utils.preprocess import crop_one_image

        crop_one_image(SimpleNamespace(img_path=args.img_path, save_path=crop_path))
        inference_img_path = crop_path

    make_bg_for_one_image(SimpleNamespace(img_path=inference_img_path, save_path=bg_path))

    inference_args = parse_inference_args(
        [
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--seed",
            str(args.seed),
            "--inference_steps",
            str(args.inference_steps),
            "--denoising_unet_path",
            args.denoising_unet_path,
            "--ID_unet_path",
            args.ID_unet_path,
            "--au_test",
            args.au_test,
            "--AU_variation",
            args.AU_variation,
            "--img_path",
            str(inference_img_path),
            "--bg_path",
            str(bg_path),
            "--saved_path",
            str(args.saved_path),
        ]
    )
    inference_args.revision = args.revision
    inference_args.variant = args.variant
    run_inference(inference_args)

    generated_path = args.saved_path / inference_img_path.name
    output_name = args.img_path.name
    if not args.img_path.suffix:
        output_name = f"{output_name}.png"
    output_path = args.saved_path / output_name
    if generated_path.exists() and generated_path != output_path:
        generated_path.replace(output_path)


def main():
    args = parse_args()
    if args.work_dir:
        run_pipeline(args, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="magicface_") as work_dir:
            run_pipeline(args, work_dir)


if __name__ == "__main__":
    main()
