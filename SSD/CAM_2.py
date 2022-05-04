import torchvision
import torch
import tqdm
import click
import numpy as np
import tops
import requests
#from torchvision import transfoms as T
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("image_dir", type=click.Path(exists=True, dir_okay=True, path_type=str))
@click.argument("output_dir", type=click.Path(dir_okay=True, path_type=str))
@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.3)



def run_demo(config_path: Path, score_threshold: float, image_dir: Path, output_dir: Path):
    score_threshold=0.5
    config_path = Path(config_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])

    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    output_dir.mkdir(exist_ok=True, parents=True)
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

    for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Predicting on images")):
        image_name = image_path.stem
        orig_img = np.array(Image.open(image_path).convert("RGB"))
        height, width = orig_img.shape[:2]
        image_float_np = (np.float32(orig_img) / 255)
        img = cpu_transform({"image": orig_img})["image"]#.unsqueeze(0)
        img = tops.to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        


        boxes, categories, scores = model(img,score_threshold=score_threshold)[0]
        boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]
        
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        drawn_image = draw_boxes(
            orig_img, boxes, categories, scores).astype(np.uint8)

        target_layers = [model.feature_extractor]
        targets = [FasterRCNNBoxScoreTarget(labels=categories, bounding_boxes=boxes)] #vategories.tolist()

    

        cam = EigenCAM(model,
               target_layers, 
               use_cuda=torch.cuda.is_available(),
               reshape_transform=fasterrcnn_reshape_transform)


        grayscale_cam = cam(img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        

        norm_cam = renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam )

        image_with_bounding_boxes = draw_boxes( norm_cam, boxes, categories, scores).astype(np.uint8)

        im = Image.fromarray(image_with_bounding_boxes)
        

        output_path = output_dir.joinpath(f"{image_name}.png")
        im.save(output_path)


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    #fix box problem with neg values
    boxes = np.int32(np.abs(boxes))
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    #image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return eigencam_image_renormalized

#Image.fromarray(renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam))


def fasterrcnn_reshape_transform(x):


    target_size = x[3].size()[-2 : ]
    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations




if __name__ == '__main__':
    run_demo()