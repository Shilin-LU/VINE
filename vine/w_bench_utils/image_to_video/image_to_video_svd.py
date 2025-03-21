import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "./generative-models/"))
import math, argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor
from typing import Optional, Union, List


def sample(
    INPUT_PATH: str = "",
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[Union[float, List[float]]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    SVDxt_OUTPUT_PATH = None,
    targetFrames = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = SVDxt_OUTPUT_PATH
        model_config = "./w_bench_utils/image_to_video/generative-models/scripts/sampling/configs/svd_xt.yaml"
        """
        elif version == "svd_image_decoder":
            num_frames = default(num_frames, 14)
            num_steps = default(num_steps, 25)
            output_folder = default(
                output_folder, "./w_bench_utils/image_to_video/generative-models/outputs/svd_image_decoder/"
            )
            model_config = "./w_bench_utils/image_to_video/generative-models/scripts/sampling/configs/svd_image_decoder.yaml"
        elif version == "svd_xt_image_decoder":
            num_frames = default(num_frames, 25)
            num_steps = default(num_steps, 30)
            output_folder = default(
                output_folder, "./w_bench_utils/image_to_video/generative-models/outputs/svd_xt_image_decoder/"
            )
            model_config = "./w_bench_utils/image_to_video/generative-models/scripts/sampling/configs/svd_xt_image_decoder.yaml"
        elif version == "sv3d_u":
            num_frames = 21
            num_steps = default(num_steps, 50)
            output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_u/")
            model_config = "./w_bench_utils/image_to_video/generative-models/scripts/sampling/configs/sv3d_u.yaml"
            cond_aug = 1e-5
        elif version == "sv3d_p":
            num_frames = 21
            num_steps = default(num_steps, 50)
            output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_p/")
            model_config = "./w_bench_utils/image_to_video/generative-models/scripts/sampling/configs/sv3d_p.yaml"
            cond_aug = 1e-5
            if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
                elevations_deg = [elevations_deg] * num_frames
            assert (
                len(elevations_deg) == num_frames
            ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
            polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
            if azimuths_deg is None:
                azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
            assert (
                len(azimuths_deg) == num_frames
            ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
            azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
            azimuths_rad[:-1].sort()
        """
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    path = Path(INPUT_PATH)
    all_img_paths = []
    if path.is_file():
        if any([INPUT_PATH.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [INPUT_PATH]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in tqdm(all_img_paths):
        input_image = None
        if "sv3d" in version:
            image = Image.open(input_img_path)
            if image.mode == "RGBA":
                pass
            else:
                # remove bg
                image.thumbnail([768, 768], Image.Resampling.LANCZOS)
                # image = remove(image.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr = np.array(image)
            in_w, in_h = image_arr.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr[y : y + h, x : x + w]
            # resize frame to 576x576
            rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))
        else:
            with Image.open(input_img_path) as image:
                if image.mode == "RGBA":
                    input_image = image.convert("RGB")
                w, h = image.size

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    input_image = input_image.resize((width, height))
                    print(
                        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )
                else:
                    input_image = image

        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        image = ToTensor()(Image.open(input_img_path))
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        if "sv3d_p" in version:
            value_dict["polars_rad"] = polars_rad
            value_dict["azimuths_rad"] = azimuths_rad

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)

                for idx, sample in enumerate(samples):
                    if not targetFrames.get(idx):
                        continue
                    image_array = sample.permute(1, 2, 0).detach().cpu().numpy()  # 从 (C, H, W) 转换为 (H, W, C)
                    image_array = np.clip(image_array, 0.0, 1.0)
                    image_array = (image_array * 255).astype(np.uint8)
                    imageio.imwrite(
                        os.path.join(output_folder, f"{'_'.join(str(Path(input_img_path)).split('/')[-1].split('_')[:2])}_{idx:04d}.png"), image_array
                    )
                """
                vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(vid.shape[2], vid.shape[1])) \
                    .output(video_path, pix_fmt='yuv420p') \
                    .run(input=vid.tobytes())
                """


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_images_folder', type=str, default="./vine_encoded_wbench/512/SVD_1K")
    parser.add_argument('--edited_output_folder', type=str, default="./output/edited_wmed_wbench/SVD_raw")
    args = parser.parse_args() 
    
# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    targetFrames = {key: 1 for key in [5, 7, 9, 11, 13, 15, 17, 19]}         # (Target Frames Control)
    
    os.makedirs(args.edited_output_folder, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------

    sample(INPUT_PATH=args.wm_images_folder, SVDxt_OUTPUT_PATH=args.edited_output_folder, targetFrames=targetFrames)
