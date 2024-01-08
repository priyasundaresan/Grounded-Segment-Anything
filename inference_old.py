import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram, ram_plus
from ram.utils import build_openset_llm_label_embedding
from torch import nn
from ram import inference_ram
import torchvision.transforms as TS


import matplotlib
matplotlib.use('agg')
from ultralytics import YOLO

from EfficientSAM.FastSAM.tools import *
from groundingdino.util.inference import predict, annotate, Model
from torchvision.ops import box_convert
import ast


# import openai
# import nltk

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def convert_image_np(image_np):
    # load image
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--better_quality",
        type=str,
        default=True,
        help="better quality using morphologyEx",
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )

    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--image_dir", type=str, default="test_images", help="path to input image folder")
    parser.add_argument('--llm_tag_des', help='path to LLM tag descriptions', default='custom_tag_descriptions_short.json')
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    
    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
        # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    #ram_model = ram(pretrained=ram_checkpoint,
    #                                    image_size=384,
    #                                    vit='swin_l')

    ram_model = ram_plus(pretrained=ram_checkpoint,
                             image_size=384,
                             vit='swin_l')
    #######set custom interference
    print('Building tag embedding:')
    with open(args.llm_tag_des, 'rb') as fo:
        llm_tag_des = json.load(fo)
    custom_label_embedding, custom_categories = build_openset_llm_label_embedding(llm_tag_des)
    ram_model.tag_list = np.array(custom_categories)
    ram_model.label_embed = nn.Parameter(custom_label_embedding.float())
    ram_model.num_class = len(custom_categories)
    # the threshold for unseen categories is often lower
    ram_model.class_threshold = torch.ones(ram_model.num_class) * 0.5
    #######

    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)

    sam_model = YOLO(args.sam_checkpoint)


    for fn in os.listdir(args.image_dir):
        img_path = os.path.join(args.image_dir, fn)
        results = sam_model(
            img_path, 
            imgsz=1024,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )

        # load image
        image_pil, image = load_image(os.path.join(args.image_dir, fn))
        size = image_pil.size
        H, W = size[1], size[0]

        raw_image = image_pil.resize(
                        (384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device)
        res = inference_ram(raw_image , ram_model)
        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags=res[0].replace(' |', ',')
        print("Image Tags: ", res[0])
        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, tags, box_threshold, text_threshold, device=device
        )
        image = cv2.imread(os.path.join(args.image_dir, fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes_filt_rescaled = boxes_filt * torch.Tensor([W, H, W, H])
        boxes_sam_format = box_convert(boxes=boxes_filt_rescaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()


        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        boxes_sam_format = boxes_sam_format[nms_idx].tolist()
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        # draw output image
        plt.figure(figsize=(10, 10))
        #plt.imshow(plate_image)
        plt.imshow(image)
        for box_idx, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
            show_box(box.numpy(), plt.gca(), label)
            mask, _ = box_prompt(
                results[0].masks.data,
                boxes_sam_format[box_idx],
                H,
                W,
            )
            annotations = np.array([mask])
            args.img_path = img_path
            img_array = fast_process(
                annotations=annotations,
                args=args,
                mask_random_color=False,
                bbox=boxes_sam_format[box_idx],
            )
            cv2.imwrite(os.path.join(output_dir, fn.replace('.jpg', f'_{label}.jpg')), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, fn), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
