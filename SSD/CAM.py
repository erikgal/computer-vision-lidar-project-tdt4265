import torchvision
import torch
import tqdm
import click
import numpy as np
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
"""
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
"""
# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("image_dir", type=click.Path(exists=True, dir_okay=True, path_type=str))
@click.argument("output_dir", type=click.Path(dir_okay=True, path_type=str))
@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.3)
def run_demo(config_path: Path, score_threshold: float, image_dir: Path, output_dir: Path):
    config_path = Path(config_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    output_dir.mkdir(exist_ok=True, parents=True)
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

    #code from https://linuxtut.com/en/082f71b96b9aca0d5df5/
    # Grad-CAM
    target_layer = model
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)

    images = []

    for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Predicting on images")):
        """
        image_name = image_path.stem
        orig_img = np.array(Image.open(image_path).convert("RGB"))
        height, width = orig_img.shape[:2]
        img = cpu_transform({"image": orig_img})["image"].unsqueeze(0)
        img = tops.to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        boxes, categories, scores = model(img,score_threshold=score_threshold)[0]
        print(scores)
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]
        drawn_image = draw_boxes(
            orig_img, boxes, categories, scores).astype(np.uint8)
        im = Image.fromarray(drawn_image)
        output_path = output_dir.joinpath(f"{image_name}.png")
        im.save(output_path)
        """

        img = Image.open(image_path)
        torch_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(img).to(device)
        normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
        
        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    grid_image = make_grid(images, nrow=5)
     #View results
    transforms.ToPILImage()(grid_image)

    """
    #Assuming you are calling a validation dataset for a label
    for path in glob.glob("{}/label1/*".format(config['dataset'])):
        img = Image.open(path)
        torch_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(img).to(device)
        normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
        
        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    grid_image = make_grid(images, nrow=5)
    """
   

if __name__ == '__main__':
    run_demo()