# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path
import os
import yaml
import time
import torch
import pprint 
import shutil
import argparse
import datetime
import warnings
import subprocess
import utils.constants as const
import utils.video_grid_utils as vgu
from pipelines.sd_controlnet_rave import RAVE
from pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet
warnings.filterwarnings("ignore")

MODEL_CACHE = "checkpoints"
SD_URL = "https://storage.googleapis.com/replicate-weights/runwayml/stable-diffusion-v1-5/ded79e2/stable-diffusion-v1-5.tar"

def init_device():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device

def init_paths(input_ns):
    if input_ns.save_folder == None or input_ns.save_folder == '':
        input_ns.save_folder = input_ns.video_name
    else:
        input_ns.save_folder = os.path.join(input_ns.save_folder, input_ns.video_name)
    save_dir = os.path.join(const.OUTPUT_PATH, input_ns.save_folder)
    os.makedirs(save_dir, exist_ok=True)
    save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
    input_ns.save_path = os.path.join(save_dir, f'{input_ns.positive_prompts}-{str(save_idx).zfill(5)}')
    
    
    if '-' in input_ns.preprocess_name:
        input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
    else:
        input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
    input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
    
    input_ns.inverse_path = os.path.join(const.GENERATED_DATA_PATH, 'inverses', input_ns.video_name, f'{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}')
    input_ns.control_path = os.path.join(const.GENERATED_DATA_PATH, 'controls', input_ns.video_name, f'{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}')
    os.makedirs(input_ns.control_path, exist_ok=True)
    os.makedirs(input_ns.inverse_path, exist_ok=True)
    os.makedirs(input_ns.save_path, exist_ok=True)
    return input_ns

def run(*args):
    batch_size = 4
    batch_size_vae = 1
    is_ddim_inversion = True
    is_shuffle = True
    num_inference_steps = 20
    num_inversion_step = 20
    cond_step_start = 0.0
    give_control_inversion = True
    inversion_prompt = ''
    save_folder = ''
    list_of_inputs = [x for x in args]
    input_ns = argparse.Namespace(**{})
    input_ns.video_path = list_of_inputs[0] # video_path 
    input_ns.video_name = os.path.basename(input_ns.video_path).replace('.mp4', '').replace('.gif', '') 
    input_ns.preprocess_name = list_of_inputs[1]
    input_ns.batch_size = batch_size
    input_ns.batch_size_vae = batch_size_vae
    input_ns.cond_step_start = cond_step_start
    input_ns.controlnet_conditioning_scale = list_of_inputs[2]  
    input_ns.controlnet_guidance_end = list_of_inputs[3]  
    input_ns.controlnet_guidance_start = list_of_inputs[4]  
    input_ns.give_control_inversion = give_control_inversion  
    input_ns.grid_size = list_of_inputs[5]  
    input_ns.sample_size = list_of_inputs[6]  
    input_ns.pad = list_of_inputs[7]
    input_ns.guidance_scale = list_of_inputs[8]
    input_ns.inversion_prompt = inversion_prompt
    input_ns.is_ddim_inversion = is_ddim_inversion
    input_ns.is_shuffle = is_shuffle
    input_ns.negative_prompts = list_of_inputs[9]  
    input_ns.num_inference_steps = num_inference_steps
    input_ns.num_inversion_step = num_inversion_step
    input_ns.positive_prompts = list_of_inputs[10] 
    input_ns.save_folder = save_folder
    input_ns.seed = list_of_inputs[11]  
    input_ns.model_id = list_of_inputs[12]
    diffusers_model_path = os.path.join(const.CWD, 'CIVIT_AI', 'diffusers_models')
    os.makedirs(diffusers_model_path, exist_ok=True)
    if 'model_id' not in list(input_ns.__dict__.keys()):
        input_ns.model_id = "None"

    device = init_device()
    input_ns = init_paths(input_ns)
    input_ns.image_pil_list = vgu.prepare_video_to_grid(input_ns.video_path, input_ns.sample_size, input_ns.grid_size, input_ns.pad)

    print(input_ns.video_path)
    input_ns.sample_size = len(input_ns.image_pil_list)
    print(f'Frame count: {len(input_ns.image_pil_list)}')

    controlnet_class = RAVE_MultiControlNet if '-' in str(input_ns.controlnet_conditioning_scale) else RAVE
    CN = controlnet_class(device)
    CN.init_models(input_ns.hf_cn_path, input_ns.hf_path, input_ns.preprocess_name, input_ns.model_id)
    
    input_dict = vars(input_ns)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(input_dict)
    yaml_dict = {k:v for k,v in input_dict.items() if k != 'image_pil_list'}

    start_time = datetime.datetime.now()
    if '-' in str(input_ns.controlnet_conditioning_scale):
        res_vid, control_vid_1, control_vid_2 = CN(input_dict)
    else: 
        res_vid, control_vid = CN(input_dict)
    end_time = datetime.datetime.now()
    save_name = f"{'-'.join(input_ns.positive_prompts.split())}_cstart-{input_ns.controlnet_guidance_start}_gs-{input_ns.guidance_scale}_pre-{'-'.join((input_ns.preprocess_name.replace('-','+').split('_')))}_cscale-{input_ns.controlnet_conditioning_scale}_grid-{input_ns.grid_size}_pad-{input_ns.pad}_model-{os.path.basename(input_ns.model_id)}"
    res_vid[0].save(os.path.join(input_ns.save_path, f'{save_name}.gif'), save_all=True, append_images=res_vid[1:], loop=10000)
    control_vid[0].save(os.path.join(input_ns.save_path, f'control_{save_name}.gif'), save_all=True, append_images=control_vid[1:], optimize=False, loop=10000)

    yaml_dict['total_time'] = (end_time - start_time).total_seconds()
    yaml_dict['total_number_of_frames'] = len(res_vid)
    yaml_dict['sec_per_frame'] = yaml_dict['total_time']/yaml_dict['total_number_of_frames']
    with open(os.path.join(input_ns.save_path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
    
    return os.path.join(input_ns.save_path, f'{save_name}.gif'), os.path.join(input_ns.save_path, f'control_{save_name}.gif')

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Output(BaseModel):
    video: Path
    control: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
        if not os.path.exists(MODEL_CACHE):
            download_weights(SD_URL, MODEL_CACHE)

    def predict(
        self,
        video: Path = Input(description="Input mp4 video (recommended size of 512x512 or 512x320)"),
        positive_prompt: str = Input(description="Positive prompts separated by spaces", default="A black panther"),
        negative_prompt: str = Input(description="Negative prompts separated by spaces", default=""),
        control_type: str = Input(
            description="Control type",
            default="depth_zoe",
            choices=[
                "lineart_releastic",
                "lineart_coart",
                "lineart_standard",
                "lineart_anime",
                "lineart_anime_denoise",
                "softedge_hed",
                "softedge_hedsafe",
                "softedge_pidinet",
                "softedge_pidsafe",
                "canny",
                "depth_leres",
                "depth_leres++",
                "depth_midas",
                "depth_zoe",
            ]
        ),
        guidance_scale: float = Input(description="Guidance scale", default=7.5, ge=0.01, le=40.0),
        seed: int = Input(description="Seed", default=0, ge=0, le=2147483647),
    ) -> Output:
        # Clear past run inputs and results, and create necessary directories
        shutil.rmtree('./data/mp4_videos', ignore_errors=True)
        os.makedirs('./data/mp4_videos')
        shutil.copy(video, './data/mp4_videos/demo.mp4')
        # shutil.rmtree('./results/cog', ignore_errors=True)
        # os.makedirs('./results/cog')

        # # Prepare inputs for run function
        output_path = run(
            './data/mp4_videos/demo.mp4',
            control_type,
            1.0,
            1.0,
            0.0,
            3,
            1,
            2,
            guidance_scale,
            negative_prompt,
            positive_prompt, 
            seed,
            'checkpoints'
        )
        # print("Output pair: ", output_path)

        return Output(video=Path(output_path[0]), control=Path(output_path[1]))
    