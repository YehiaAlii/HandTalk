import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_online
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

_GLOBAL_MODEL = None

class InferenceConfig:
    def __init__(self):
        # Essential parameters for inference
        self.seed = 42
        self.finetune = "./out/official_checkpoints/official_openasl_pose_only_slt.pth"
        self.task = "SLT"
        self.dataset = "WLASL"
        self.output_dir = "./output"
        
        # Model parameters
        self.batch_size = 1
        self.hidden_dim = 256
        self.max_length = 256
        self.rgb_support = False
        
        # Required for model initialization (but not used in inference)
        self.label_smoothing = 0.2
        self.eval = True
        self.num_workers = 2
        self.pin_mem = True

def get_model(config=None):
    """Get or initialize the global model"""
    global _GLOBAL_MODEL
    
    if _GLOBAL_MODEL is None:
        if config is None:
            config = InferenceConfig()
        print("Loading model into global cache...")
        _GLOBAL_MODEL = prepare_model(config)
        print("Model successfully loaded into global cache")
    
    return _GLOBAL_MODEL

def prepare_model(config):
    """Initialize and load the model with pre-trained weights"""
    print("Creating model...")
    model = Uni_Sign(args=config)
    model.cuda()
    model.train()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if config.finetune:
        print('***********************************')
        print('Loading checkpoint...')
        print('***********************************')
        state_dict = torch.load(config.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    else:
        raise ValueError("A pre-trained model path must be provided")

    model.eval()
    model.to(torch.bfloat16)
    return model

def process_video_for_inference(video_path, config, pre_extracted_keypoints=None):
    """Process a video file and prepare it for inference"""
    
    if pre_extracted_keypoints is None:
        from preprocessing import extract_keypoints_from_video
        
        # Extract keypoints from the video
        pose_data = extract_keypoints_from_video(
            video_path, 
            device='cuda', 
            backend='onnxruntime', 
            max_workers=4
        )
    else:
        pose_data = pre_extracted_keypoints
        print("Using pre-extracted keypoints")
    
    # Create dataset
    print("Creating dataset...")
    online_data = S2T_Dataset_online(args=config)
    online_data.rgb_data = video_path
    online_data.pose_data = pose_data
    
    # Create dataloader
    online_sampler = torch.utils.data.SequentialSampler(online_data)
    online_dataloader = DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=online_sampler,
    )
    
    return online_dataloader

def run_inference(model, data_loader):
    """Run inference with the model on the provided data"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype is not None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            stack_out = model(src_input, tgt_input)

            output = model.generate(
                stack_out,
                max_new_tokens=100,
                num_beams=4,
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])

    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    return tgt_pres[0]

def translate_sign_language(video_path, config=None, pre_extracted_keypoints=None):
    """
    End-to-end function to translate sign language from a video
    
    Args:
        video_path: Path to the video file
        config: Configuration object
        pre_extracted_keypoints: Pre-extracted keypoints to avoid duplicate extraction
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use default config if none provided
    if config is None:
        config = InferenceConfig()
    
    # Ensure output directory exists
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    utils.set_seed(config.seed)
    
    # Prepare the model
    model = get_model(config)
    
    # Process the video and prepare data for inference
    data_loader = process_video_for_inference(video_path, config, pre_extracted_keypoints)
    
    # Run inference
    prediction = run_inference(model, data_loader)
    
    print(f"Prediction result: {prediction}")
    return prediction