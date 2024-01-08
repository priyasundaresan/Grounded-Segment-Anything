import argparse
import cv2
import matplotlib
matplotlib.use('agg')
from ultralytics import YOLO
from FastSAM.tools import *
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert
import ast

import time
from threading import Lock

# ros imports
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import rospy

class InferenceServer:
    def __init__(self):
        self.model_path = './FastSAM/FastSAM-x.pt'
        self.img_path = '/tmp/image.png'
        #self.text = 'the green-colored pear, not a lemon.'
        self.text = 'the blue plate with food on it.'
        self.imgsz = 1024
        self.iou = 0.9
        self.conf = 0.4
        self.output = './output/'
        self.randomcolor = True
        self.point_prompt = "[[0,0]]"
        self.point_label = "[0]"
        self.box_prompt = "[0,0,0,0]"
        self.better_quality = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retina = True
        self.withContours = False

    def run_inference(self):
        img_path = self.img_path
        text = self.text

        # path to save img
        save_path = self.output
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #basename = os.path.basename(self.img_path).split(".")[0]

        # Build Fast-SAM Model
        model = YOLO(self.model_path)

        results = model(
            self.img_path,
            imgsz=self.imgsz,
            device=self.device,
            retina_masks=self.retina,
            iou=self.iou,
            conf=self.conf,
            max_det=100,
        )

        # Build GroundingDINO Model
        groundingdino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_ckpt_path = "./groundingdino_swint_ogc.pth"

        image_source, image = load_image(img_path)
        model = load_model(groundingdino_config, groundingdino_ckpt_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text,
            box_threshold=0.2,
            text_threshold=0.25,
            device=self.device,
        )


        # Grounded-Fast-SAM

        ori_img = cv2.imread(img_path)
        ori_h = ori_img.shape[0]
        ori_w = ori_img.shape[1]

        # Save each frame due to the post process from FastSAM
        boxes = boxes * torch.Tensor([ori_w, ori_h, ori_w, ori_h])
        print(f"Detected Boxes: {len(boxes)}")
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()
        for box_idx in range(len(boxes)):
            mask, _ = box_prompt(
                results[0].masks.data,
                boxes[box_idx],
                ori_h,
                ori_w,
            )
            annotations = np.array([mask])
            img_array = fast_process(
                annotations=annotations,
                args=self,
                mask_random_color=True,
                bbox=boxes[box_idx],
            )
            #cv2.imwrite(os.path.join(save_path, basename + f"_{str(box_idx)}_caption_{phrases[box_idx]}.jpg"), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            cv2.imshow('img', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

class RealSenseROS:
    def __init__(self):
        self.bridge = CvBridge()

        self.camera_lock = Lock()
        self.camera_header = None
        self.camera_color_data = None
        self.camera_info_data = None
        self.camera_depth_data = None

        queue_size = 1000

        self.color_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size= queue_size, buff_size = 65536*queue_size)
        self.camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo, queue_size= queue_size, buff_size = 65536*queue_size)
        self.depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size= queue_size, buff_size = 65536*queue_size)
        ts_top = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], queue_size= queue_size)
        ts_top.registerCallback(self.rgbdCallback)
        ts_top.enable_reset = True

        time.sleep(1.0)

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
        

if __name__ == "__main__":
    rospy.init_node('RealSenseROS')
    rs_ros = RealSenseROS()
    inference_server = InferenceServer()
    header, color_data, info_data, depth_data = rs_ros.get_camera_data()
    cv2.imwrite('/tmp/image.png', color_data)
    inference_server.run_inference()

    #cv2.imshow('img', color_data)
    #cv2.waitKey(0)


    ## save data to file
    #import os
    #import cv2
    #file = 'output'
    #num_files = len([name for name in os.listdir(file)])
    #cv2.imwrite(file + str(num_files) + "_camera_color_data.jpg", color_data)

    #print("Header:",header)
    #print("Color Data:",color_data[0:10])
    #print("Info Data:",info_data)
    #print("Depth Data:",depth_data[0:10])
