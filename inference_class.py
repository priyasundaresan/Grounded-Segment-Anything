import cv2
import time
import os
import numpy as np
import supervision as sv

import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model
from vision_utils import *

import os
from openai import OpenAI
import ast

class BiteAcquisitionInference:
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # GroundingDINO config and checkpoint
        self.GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
        
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        
        # Building MobileSAM predictor
        self.EFFICIENT_SAM_CHECKPOINT_PATH = "efficientsam_s_gpu.jit"
        self.efficientsam = torch.jit.load(self.EFFICIENT_SAM_CHECKPOINT_PATH)

        self.FOOD_CLASSES = ["noodles", "meat", "shrimp", "chicken", "vegetable", "broccoli"]
        self.BOX_THRESHOLD = 0.23
        self.TEXT_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.65

        self.CATEGORIES = ['Meat/Seafood', 'Vegetable', 'Noodles', 'Fruit', 'Sweet Dip', 'Savory Dip']

        self.api_key = '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'
        self.client = OpenAI(api_key=self.api_key)

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

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):
        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.camera_lock:
            self.camera_header = rgb_image_msg.header
            self.camera_color_data = rgb_image
            self.camera_info_data = camera_info_msg
            self.camera_depth_data = depth_image

    def get_camera_data(self):
        with self.camera_lock:
            return self.camera_header, self.camera_color_data, self.camera_info_data, self.camera_depth_data

    def detect_items(self, image):
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

        noodle_mask = individual_masks[0]
        noodle_mask = cv2.bitwise_and(cv2.bitwise_not(blue_mask), noodle_mask)
        noodle_mask = outpaint_masks(noodle_mask, individual_masks[1:])
        individual_masks[0] = noodle_mask
        individual_masks = [cleanup_mask(mask) for mask in individual_masks]

        densest = detect_densest(noodle_mask)
        sparsest = detect_sparsest(noodle_mask, densest)

        visualized_masks = []
        #visualized_masks = [blue_mask]
        for i, mask in enumerate(individual_masks):
            if 'noodle' in labels[i]:
                vis = visualize_keypoints(mask.copy(), [densest, sparsest])
            else:
                centroid = detect_centroid(mask)
                vis = visualize_keypoints(mask.copy(), [centroid])
            visualized_masks.append(vis)

        #return annotated_image, individual_masks, labels
        return annotated_image, visualized_masks, labels

    def categorize_items(self, labels):
        food_item_count = {c:0 for c in self.CATEGORIES}
        prompt = """
                 Input: 'noodles 0.69'
                 Output: 'Noodles'

                 Input: 'shrimp 0.26'
                 Output: 'Meat/Seafood'

                 Input: 'meat 0.46'
                 Output: 'Meat/Seafood'

                 Input: 'broccoli 0.42'
                 Output: 'Vegetable'

                 Input: 'celery 0.69'
                 Output: 'Vegetable'

                 Input: 'chicken 0.27'
                 Output: 'Meat/Seafood'

                 Input: 'ketchup 0.47'
                 Output: 'Savory Dip'

                 Input: 'ranch 0.24'
                 Output: 'Savory Dip'

                 Input: 'caramel 0.28'
                 Output: 'Sweet Dip'

                 Input: 'chocolate sauce 0.24'
                 Output: 'Sweet Dip'

                 Input: 'strawberry 0.57'
                 Output: 'Fruit'

                 Input: 'blueberry 0.87'
                 Output: 'Fruit'

                 Input: '%s'
                 Output:
                 """
        for label in labels:
            predicted_category = self.chat_with_openai(prompt%label).strip().replace("'",'')
            food_item_count[predicted_category] += 1

        #for label in labels:
        #    request = Request(model="openai/text-davinci-003", prompt=prompt%label, random=0)
        #    request_result = self.service.make_request(self.auth, request)
        #    completion = request_result.completions[0].text.strip()
        #    print(label, completion)

        #print(food_item_count)
        return food_item_count

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
    inference_server.score_bites_preference()

    #SOURCE_IMAGE_DIR = 'test_images'
    #OUTPUT_DIR = 'outputs'

    #if not os.path.exists(OUTPUT_DIR):
    #    os.mkdir(OUTPUT_DIR)

    #for i, fn in enumerate(os.listdir(SOURCE_IMAGE_DIR)):
    #    SOURCE_IMAGE_PATH = os.path.join(SOURCE_IMAGE_DIR, fn)
    #    image = cv2.imread(SOURCE_IMAGE_PATH)
    #    annotated_image, masks, labels = inference_server.detect_items(image)
    #    #inference_server.categorize_items(labels)
    #    for j in range(len(masks)):
    #        cv2.imwrite(os.path.join(OUTPUT_DIR, '%02d_%s.jpg'%(i,labels[j])), masks[j])
    #        cv2.imwrite(os.path.join(OUTPUT_DIR, '%02d_all.jpg'%(i)), annotated_image)
