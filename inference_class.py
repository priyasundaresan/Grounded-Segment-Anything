import cv2
import time
import os
import numpy as np
import supervision as sv

import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model

from vision_utils import detect_densest, detect_sparsest, detect_centroid, efficient_sam_box_prompt_segment, outpaint_masks, detect_blue, proj_pix2mask, cleanup_mask, visualize_keypoints, visualize_push, detect_plate, mask_weight, nearest_neighbor

import os
from openai import OpenAI
import ast

import base64
import requests
import cmath
import math

from src.food_pos_ori_net.model.minispanet import MiniSPANet
from src.spaghetti_segmentation.model import SegModel
import torchvision.transforms as transforms

print('imports done')

class GPT4Vision:
    def __init__(self, api_key, prompt_dir):

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        self.prompt_dir = prompt_dir
        
        with open("%s/prompt.txt"%self.prompt_dir, 'r') as f:
            self.prompt_text = f.read()
        
        self.detection_prompt_img1 = cv2.imread("%s/detection/11.jpg"%self.prompt_dir)
        self.detection_prompt_img2 = cv2.imread("%s/detection/12.jpg"%self.prompt_dir)
        self.detection_prompt_img3 = cv2.imread("%s/detection/13.jpg"%self.prompt_dir)

        self.detection_prompt_img1 = self.encode_image(self.detection_prompt_img1)
        self.detection_prompt_img2 = self.encode_image(self.detection_prompt_img2)
        self.detection_prompt_img3 = self.encode_image(self.detection_prompt_img3)

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def prompt(self, image):
        
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.prompt_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img1}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img2}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img3}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]

        return response_text

class BiteAcquisitionInference:
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GroundingDINO config and checkpoint
        self.GROUNDING_DINO_CONFIG_PATH = "/scr/priyasun/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = "/scr/priyasun/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        
        # Building MobileSAM predictor
        self.EFFICIENT_SAM_CHECKPOINT_PATH = "/scr/priyasun/Grounded-Segment-Anything/efficientsam_s_gpu.jit"
        self.efficientsam = torch.jit.load(self.EFFICIENT_SAM_CHECKPOINT_PATH)

        self.FOOD_CLASSES = ["noodles", "meatball", "shrimp", "chicken", "vegetable", "broccoli"]
        self.BOX_THRESHOLD = 0.22
        self.TEXT_THRESHOLD = 0.2
        #self.NMS_THRESHOLD = 0.65
        self.NMS_THRESHOLD = 0.6

        self.CATEGORIES = ['meat/seafood', 'vegetable', 'noodles', 'fruit', 'dip', 'plate']

        self.api_key = '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'

        self.gpt4v_client = GPT4Vision(self.api_key, '/scr/priyasun/Grounded-Segment-Anything/prompt')
        self.client = OpenAI(api_key=self.api_key)

        torch.set_flush_denormal(True)
        checkpoint_dir = 'spaghetti_checkpoints'

        self.minispanet = MiniSPANet(out_features=1)
        self.minispanet_crop_size = 100
        checkpoint = torch.load('%s/spaghetti_ori_net.pth'%checkpoint_dir, map_location=self.DEVICE)
        self.minispanet.load_state_dict(checkpoint)
        self.minispanet.eval()
        self.minispanet_transform = transforms.Compose([transforms.ToTensor()])

        self.seg_net = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)
        ckpt = torch.load('%s/spaghetti_seg_resnet.pth'%checkpoint_dir, map_location=self.DEVICE)
        self.seg_net.load_state_dict(ckpt)
        self.seg_net.eval()
        self.seg_net.to(self.DEVICE)
        self.seg_net_transform = transforms.Compose([transforms.ToTensor()])

    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items

    def chat_with_openai(self, prompt):
        """
        Sends the prompt to OpenAI API using the chat interface and gets the model's response.
        """
        message = {
                    'role': 'user',
                    'content': prompt
                  }
    
        response = self.client.chat.completions.create(
                   model='gpt-3.5-turbo-1106',
                   messages=[message]
                  )
        
        # Extract the chatbot's message from the response.
        # Assuming there's at least one response and taking the last one as the chatbot's reply.
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

    def run_minispanet_inference(self, u, v, cv_img, crop_dim=15):
        cv_crop = cv_img[v-crop_dim:v+crop_dim, u-crop_dim:u+crop_dim]
        cv_crop_resized = cv2.resize(cv_crop, (self.minispanet_crop_size, self.minispanet_crop_size))
        rescale_factor = cv_crop.shape[0]/self.minispanet_crop_size

        img_t = self.minispanet_transform(cv_crop_resized)
        img_t = img_t.unsqueeze(0)
        H,W = self.minispanet_crop_size, self.minispanet_crop_size

        heatmap, pred = self.minispanet(img_t)

        heatmap = heatmap.detach().cpu().numpy()
        pred_rot = pred.detach().cpu().numpy().squeeze()

        heatmap = heatmap[0][0]
        pred_x, pred_y = self.minispanet_crop_size//2, self.minispanet_crop_size//2
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(cv_crop_resized, 0.55, heatmap, 0.45, 0)
        cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
        cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
        pt = cmath.rect(20, np.pi/2-pred_rot)  
        x2 = int(pt.real)
        y2 = int(pt.imag)
        rot_vis = cv2.line(cv_crop_resized, (pred_x-x2,pred_y+y2), (pred_x+x2, pred_y-y2), (255,255,255), 2)
        cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(rot_vis,"Skewer Angle",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.circle(rot_vis, (pred_x,pred_y), 4, (255,255,255), -1)
        result = rot_vis

        global_x = u + pred_x*rescale_factor
        global_y = v + pred_y*rescale_factor
        pred_rot = math.degrees(pred_rot)
        return pred_rot, int(global_x), int(global_y), result

    def check_noodle_action_validity(self, image, sparsest, densest, filling_push_start, filling_push_end):
        if filling_push_start is None and filling_push_end is None:
            return ['Twirl', 'Group']
        H,W,C = image.shape
        vis = np.zeros((H,W))

        filling_push_mask = visualize_keypoints(vis.copy(), [filling_push_start], radius=12)
        group_mask = visualize_push(vis.copy(), sparsest, densest)
        twirl_mask = visualize_keypoints(vis.copy(), [densest], radius=20)

        valid_actions = ['Push Filling']
        if not (np.any(cv2.bitwise_and(filling_push_mask, twirl_mask))):
            valid_actions.append('Twirl')
        if not (np.any(cv2.bitwise_and(filling_push_mask, group_mask))):
            valid_actions.append('Group')

        #cv2.imshow('vis', np.hstack((filling_push_mask, group_mask, twirl_mask)))
        #cv2.waitKey(0)
        return valid_actions

    def get_noodle_action(self, image, masks, labels, categories):
        # Extract mask of only noodles
        H,W,C = image.shape
        plate_mask = masks[-1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp = self.seg_net_transform(img_rgb).to(device=self.DEVICE)
        logits = self.seg_net(inp)
        pr_mask = logits.sigmoid().detach().cpu().numpy().reshape(H,W,1)
        noodle_vapors_mask = cv2.normalize(pr_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        noodle_idx = None
        noodle_mask = None
        for i, (label, mask) in enumerate(zip(labels, masks)):
            if 'noodle' in labels[i] or 'fettucine' in labels[i] or 'spaghetti' in labels[i]:
                noodle_idx = i
                noodle_mask = mask
        noodle_mask = cv2.bitwise_and(plate_mask, noodle_mask)
        noodle_mask = cv2.bitwise_and(noodle_vapors_mask, noodle_mask)
        noodle_mask = outpaint_masks(noodle_mask.copy(), masks[:noodle_idx] + masks[noodle_idx+1:-1])

        # Detect densest and furthest points
        densest = detect_densest(noodle_mask)
        sparsest, sparsest_candidates = detect_sparsest(noodle_mask, densest)

        # Detect twirl angle
        twirl_angle, _, _, minispanet_vis = self.run_minispanet_inference(densest[0], sparsest[1], image)

        filling_centroids = []
        filling_push_start = None
        filling_push_end = None

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category in ['meat/seafood', 'vegetable']:
                centroid = detect_centroid(masks[i])
                filling_centroids.append(centroid)

        vis = image.copy()
        if len(filling_centroids):
            nearest_filling_centroid = nearest_neighbor(filling_centroids, densest)
            #filling_push_end = nearest_neighbor(sparsest_candidates, filling_push_start)
            direction = np.array(sparsest) - np.array(densest)
            direction = direction / np.linalg.norm(direction)
            direction = np.array([-direction[1], direction[0]])
            direction = (50*(direction)).astype(int)
            #filling_push_end = filling_push_start + direction

            filling_push_end = nearest_neighbor(sparsest_candidates, nearest_filling_centroid + direction)
            offset = filling_push_end - nearest_filling_centroid
            offset = 30*(offset / np.linalg.norm(offset))
            filling_push_start = np.array(nearest_filling_centroid - offset).astype(int)

            vis = visualize_push(vis, filling_push_start, filling_push_end)
        vis = visualize_push(vis, sparsest, densest)

        valid_actions = self.check_noodle_action_validity(image, sparsest, densest, nearest_filling_centroid, filling_push_end)
        #print(valid_actions)
        cv2.imshow('img', vis)
        cv2.waitKey(0)
        #cv2.imshow('img', minispanet_vis)
        #cv2.waitKey(0)
        
        return densest, sparsest, noodle_mask, valid_actions

    def detect_items(self, image):
        self.FOOD_CLASSES = [f.replace('fettucine', 'noodles') for f in self.FOOD_CLASSES]
        self.FOOD_CLASSES = [f.replace('spaghetti', 'noodles') for f in self.FOOD_CLASSES]

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.FOOD_CLASSES,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{self.FOOD_CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.NMS_THRESHOLD
        ).numpy().tolist()
        
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        
        print(f"After NMS: {len(detections.xyxy)} boxes")

        # collect segment results from EfficientSAM
        result_masks = []
        for box in detections.xyxy:
            mask = efficient_sam_box_prompt_segment(image, box, self.efficientsam)
            result_masks.append(mask)
        
        detections.mask = np.array(result_masks)
        
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{self.FOOD_CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        blue_mask = detect_blue(image.copy())

        individual_masks = []

        #print(self.FOOD_CLASSES)
        for i in range(len(detections)):
            mask_annotator = sv.MaskAnnotator(color=sv.Color.white())
            H,W,C = image.shape
            mask = np.zeros_like(image).astype(np.uint8)
            d = sv.Detections(xyxy=detections.xyxy[i].reshape(1,4), \
                              mask=detections.mask[i].reshape((1,H,W)), \
                              class_id = np.array(detections.class_id[i]).reshape((1,)))
            mask = mask_annotator.annotate(scene=mask, detections=d)
            binary_mask = np.zeros((H,W)).astype(np.uint8)
            ys,xs,_ = np.where(mask > (0,0,0))
            binary_mask[ys,xs] = 255
            individual_masks.append(binary_mask)
            #if 'noodle' in labels[i] or 'fettucine' in labels[i] or 'spaghetti' in labels[i]:
            #    noodle_idx = i
            #    noodle_mask = binary_mask

        plate_mask = detect_plate(image)

        #if noodle_mask is not None:
        #    other_masks = individual_masks[:noodle_idx] + individual_masks[noodle_idx+1:])
        #    densest, sparsest, noodle_mask = self.get_noodle_action(image, noodle_mask, other_masks, plate_mask)
        #    individual_masks[noodle_idx] = noodle_mask

        individual_masks.append(plate_mask)
        labels.append('blue plate')

        refined_masks = []

        #portion_weights = []
        for mask in individual_masks:
            refined_masks.append(cleanup_mask(mask))
        #    portion_weights.append(mask_weight(mask))

        #visualized_masks = []
        #for i, mask in enumerate(refined_masks):
        #    if 'noodle' in labels[i]:
        #        vis = visualize_keypoints(mask.copy(), [densest, sparsest])
        #    else:
        #        centroid = detect_centroid(mask)
        #        vis = visualize_keypoints(mask.copy(), [centroid])
        #    visualized_masks.append(vis)

        #keypoints = [sparsest, densest]
        #min_weight = min(portion_weights)
        #portion_weights = [p/min_weight for p in portion_weights]

        #return annotated_image, visualized_masks, labels, keypoints
        return annotated_image, refined_masks, labels

    def categorize_items(self, labels):
        categories = []
        prompt = """
                 Input: 'noodles 0.69'
                 Output: 'noodles'

                 Input: 'shrimp 0.26'
                 Output: 'meat/seafood'

                 Input: 'meat 0.46'
                 Output: 'meat/seafood'

                 Input: 'broccoli 0.42'
                 Output: 'vegetable'

                 Input: 'celery 0.69'
                 Output: 'vegetable'

                 Input: 'chicken 0.27'
                 Output: 'meat/seafood'

                 Input: 'ketchup 0.47'
                 Output: 'dip'

                 Input: 'ranch 0.24'
                 Output: 'dip'

                 Input: 'caramel 0.28'
                 Output: 'dip'

                 Input: 'chocolate sauce 0.24'
                 Output: 'dip'

                 Input: 'strawberry 0.57'
                 Output: 'fruit'

                 Input: 'blue plate'
                 Output: 'plate'

                 Input: 'blue plate'
                 Output: 'plate'

                 Input: 'blueberry 0.87'
                 Output: 'fruit'

                 Input: '%s'
                 Output:
                 """
        for label in labels:
            predicted_category = self.chat_with_openai(prompt%label).strip().replace("'",'')
            categories.append(predicted_category)

        return categories

    def score_bites_preference(self):
        prompt = """
                 Available Bites: [Noodles, Meat/Seafood, Vegetable]

                 Preference: Feed me alternating bites of spaghetti and meatballs
                 History of bites: [Noodles, Meat/Seafood, Noodles, Meat/Seafood, Noodles]
                 Preference weight, probability of next bite: 1.0, {'Meat/Seafood': 1.0, 'Noodles': 0.0, 'Vegetable': 0.0}

                 Preference: If possible, feed me alternating bites of spaghetti and meatballs
                 History of bites: [Noodles, Meat/Seafood, Noodles, Meat/Seafood]
                 Preference weight, probability of next bite: 0.7, {'Meat/Seafood': 0.0, 'Noodles': 1.0, 'Vegetable': 0.0}

                 Preference: Try to feed me only broccoli
                 History of bites: []
                 Preference weight, probability of next bite: 0.7, {'Meat/Seafood': 0.0, 'Noodles': 0.0, 'Vegetable': 1.0}

                 Preference: I slightly prefer noodles
                 History of bites: [Noodles, Vegetable, Noodles, Noodles, Meat/Seafood]
                 Preference weight, probability of next bite: 0.6, {'Meat/Seafood': 0.1, 'Noodles': 0.8, 'Vegetable': 0.1}

                 Preference: Anything is fine
                 History of bites: [Meat/Seafood, Noodles, Vegetable]
                 Preference weight, probability of next bite: 0.0, {'Meat/Seafood': 0.33, 'Noodles': 0.33, 'Vegetable': 0.33}

                 Preference: %s
                 History of bites: %s
                 Preference weight, probability of next bite:
                 """

        #preference = 'Feed me only meatballs'
        #history = '[Meat/Seafood, Meat/Seafood]'

        #preference = 'I slightly prefer meat'
        #history = '[Meat/Seafood, Meat/Seafood]'

        preference = 'I kinda want to eat some veggies'
        history = '[]'

        chatbot_response = self.chat_with_openai(prompt%(preference, history))
        weight, probabilities = ast.literal_eval(chatbot_response)
        print(weight, probabilities)

if __name__ == '__main__':
    inference_server = BiteAcquisitionInference()

    SOURCE_IMAGE_DIR = 'test_images'
    OUTPUT_DIR = 'outputs'

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    for i, fn in enumerate(os.listdir(SOURCE_IMAGE_DIR)):
        SOURCE_IMAGE_PATH = os.path.join(SOURCE_IMAGE_DIR, fn)
        image = cv2.imread(SOURCE_IMAGE_PATH)

        #items = inference_server.recognize_items(image)
        #inference_server.FOOD_CLASSES = items

        annotated_image, masks, labels = inference_server.detect_items(image)
        categories = inference_server.categorize_items(labels)

        #print(labels)
        #print(categories)
        inference_server.get_noodle_action(image, masks, labels, categories)

        if i > 3:
            break

        #cv2.imwrite(os.path.join(OUTPUT_DIR, '%02d_all.jpg'%(i)), annotated_image)
        #for j in range(len(masks)):
        #    cv2.imwrite(os.path.join(OUTPUT_DIR, '%02d_%s.jpg'%(i,labels[j])), masks[j])
