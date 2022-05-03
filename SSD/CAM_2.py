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
        print("Actual labels: ", categories)
        targets = [FasterRCNNBoxScoreTarget(labels=categories, bounding_boxes=boxes)] #vategories.tolist()
        print("Targets: ", targets)
    

        cam = EigenCAM(model,
               target_layers, 
               use_cuda=torch.cuda.is_available(),
               reshape_transform=fasterrcnn_reshape_transform)


        grayscale_cam = cam(img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

        image_with_bounding_boxes = draw_boxes( cam_image, boxes, categories, scores).astype(np.uint8)
        

        im = Image.fromarray(image_with_bounding_boxes)
        output_path = output_dir.joinpath(f"{image_name}.png")
        im.save(output_path)

        """
        grayscale_cam = cam(img, targets=targets)
        # Take the first image in the batch:
        #grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
        # And lets draw the boxes again:
        image_with_bounding_boxes = draw_boxes(orig_img, boxes, categories, scores)
        im = Image.fromarray(image_with_bounding_boxes)
        """
"""
def predict(input_tensor, model, detection_threshold):
    #print("model resp:", model(input_tensor,  score_threshold=detection_threshold))
    #print("model resp:", model(input_tensor,   score_threshold=detection_threshold))[1]
   # print("model resp:", model(input_tensor,   score_threshold=detection_threshold))[2]
    pred_boxes, pred_categories, pred_scores = model(input_tensor,  score_threshold=detection_threshold)[0]
    pred_boxes, pred_categories, pred_scores = [_.cpu().numpy() for _ in [pred_boxes, pred_categories, pred_scores]]

    
    boxes, categories, indices = [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_boxes[index].astype(np.int32))
            categories.append(pred_categories[index])
            indices.append(index)
    boxes = np.int32(boxes)
    print("boxes: ", boxes, ", cat: ", categories, " ind: ", indices)
    return boxes, categories, indices
"""
def fasterrcnn_reshape_transform(x):
    #print("\n\ninput x:", x, "with type:", type(x))
    #x = torch.tensor(x)

    target_size = x[0].size()[-2 : ]
    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    print("activations", activations, "with size:", activations.size, "--"*100)
    return activations


"""
class FasterRCNNBoxScoreTarget:
     For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
            print("output score:",output,"label", label)
        return output
"""

if __name__ == '__main__':
    run_demo()