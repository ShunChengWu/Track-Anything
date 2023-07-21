import argparse
from collections import defaultdict
import math
import os
import pathlib
import numpy as np
import requests
import torch
from tqdm import tqdm
from colmap_io import read_model
from colmap_io import BaseImage as ColmapImg
import cv2

from tools.interact_tools import SamControler 


class Image():
    def __init__(self,idx:int, image:np.array):
        self.idx = idx
        self.image = image

class ColmapTrackerSegmentAnything():
    def __init__(self, sam_checkpoint, path_colmap, args):
        '''load colmap'''
        self.reset_colmap(path_colmap)
        self.reset_sam(sam_checkpoint,args)
        
    def reset_colmap(self, path:str):
        _ ,self.imgs, pts = read_model(path)
        self.pt_scores = {k:0 for k in pts}
        # get incremental ids
        self.img_ids = sorted([img.id for img in self.imgs.values()])
        
    def reset_sam(self, path:str, args):
        self.samcontroler = SamControler(path, args.sam_model_type, args.device)
        
    def first_frame_click(self, image: Image, points:np.ndarray, labels: np.ndarray, multimask=True):
        self.samcontroler.sam_controler.reset_image()
        self.samcontroler.sam_controler.set_image(image.image)
        
        mask, logit, painted_image = self.samcontroler.first_frame_click(image.image, points, labels, multimask)
        tracked_2D_pts, tracked_3D_pts = self.get_tracked_pts(image.idx)
        self.update_score(mask, tracked_2D_pts, tracked_3D_pts)
        return mask, logit, Image(image.idx,painted_image)
    
    def generator(self, images: list, multimask:bool=True):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            torch.cuda.empty_cache()
            
            image = images[i]
            
            # Estimate threshold
            non_zero_scores = np.array([score for score in self.pt_scores.values() if score != 0])
            score_std = np.std(np.abs(non_zero_scores))
            threshold = np.abs(non_zero_scores).mean()
            if score_std == 0 or threshold == 0: 
                raise RuntimeError("No first input provided!")
            print('threshold: {} std: {}'.format(threshold,score_std))

            # Get tracked keypoints
            tracked_2D_pts, tracked_3D_pts = self.get_tracked_pts(image.idx)
            
            '''Generate promts'''
            while True:
                points = []
                labels = []
                
                for pt_id, img_xy in zip(tracked_3D_pts,tracked_2D_pts):
                    score = self.pt_scores[pt_id]
                    
                    #TODO: maybe only filter positive?
                    # if score == 0: continue
                    if score >=0:
                        if abs(score) < threshold: continue 
                    
                    label = 1 if score > 0 else -1
                    point = [img_xy[0],img_xy[1]]
                    
                    points.append(point)
                    labels.append(label)
                if len(points) == 0:
                    # reduce threshold
                    threshold -= score_std
                    print('reduce threshold to {}'.format(threshold))
                else:
                    break
                
            self.samcontroler.sam_controler.reset_image()
            self.samcontroler.sam_controler.set_image(frame)
            mask, logit, painted_image = self.samcontroler.first_frame_click(
                image.image, 
                np.array(points), 
                np.array(labels), 
                multimask)
            
            # TRY mask input
            prompts = {
                'point_coords': np.array(points),
                'point_labels': np.array(labels),
                'mask_input': logit[None,:,:]
            }
            mask, logit, painted_image = self.samcontroler.predict(
                image.image,
                prompts,
                multimask=True)
            
            self.update_score(mask, tracked_2D_pts, tracked_3D_pts)
            
            masks.append(mask)
            logits.append(logit)
            painted_images.append(Image(image.idx,painted_image))
            
        return masks, logits, painted_images
    
    def get_tracked_pts(self, idx):
        img = self.imgs[idx]
        # Get tracked keypoints
        filter_idx = np.where(img.point3D_ids>=0)
        tracked_2D_pts = img.xys[filter_idx].round().astype(int)
        tracked_3D_pts = img.point3D_ids[filter_idx]
        return tracked_2D_pts, tracked_3D_pts
    
    def update_score(self, mask, tracked_2D_pts, tracked_3D_pts):
        # Update score
        mask_score = mask[tracked_2D_pts[:,1],tracked_2D_pts[:,0]] #
        for pt_id, score in zip(tracked_3D_pts, mask_score):
            self.pt_scores[pt_id] += 1 if score == True else -1
            

def drawKeyPoints(colmap_img: ColmapImg, image_path:str):
    image_name = os.path.join(image_path,"{0:04d}.png".format(colmap_img.id))
    img = cv2.imread(image_name)
    
    imagegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kpts = [cv2.KeyPoint(x=xy[0],y=xy[1],size=1) for xy in colmap_img.xys]
    output_img = cv2.drawKeypoints(imagegray, kpts, 0, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    return output_img    

def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default='test_colmap_track_2')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 


if __name__ == '__main__':
    """
    Try to improve Tracking for SAM.
    The idea is to use the tracks from colmap. 
    All keypoints belong to the same track should have the same segmentation. 
    Here are the steps:
    1. first pos/neg clicks on one image. 
    2. Generate segmentation.
    3. Group keypoint tracks based on the segmentation.
    4. Generate clicks
    
    """
    path_colmap = '/home/sc/data/HLoc_s0.5fps2/colmap_text'
    path_images = '/home/sc/data/s0.5fps2/images_resized'
    args = parse_augment()
    
    path_out = os.path.join('result',args.exp_name)
    pathlib.Path(path_out).mkdir(exist_ok=True,parents=True)
    
    '''load colmap'''
    # cams ,imgs, pts = read_model(path_colmap)
    # pt_scores = {k:0 for k in pts}
    # # get incremental ids
    # img_ids = sorted([img.id for img in imgs.values()])
    
    # Show image and their keypoints
    # img = drawKeyPoints(imgs[1],path_images)
    # cv2.imwrite("tmp_img.png",img)
    
    '''load sam'''
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    folder ="./checkpoints"
    sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
    sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    
    ctsam = ColmapTrackerSegmentAnything(SAM_checkpoint,path_colmap,args)
    
    # samcontroler = SamControler(SAM_checkpoint, args.sam_model_type, args.device)
    multimask = "True"


    images = []
    for img_id in tqdm(ctsam.img_ids,desc="Load images to buffer"):
        img = ctsam.imgs[img_id]
        # Load image
        image_name = os.path.join(path_images,"{0:04d}.png".format(img.id))
        frame = cv2.imread(image_name, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image(img_id,frame))
        
        if len(images)>2:break
        
    # First click
    points = [
        # [int(frame.shape[0]/2), int(frame.shape[1]/2)],
        [ 100, 500],
        
        [ 100, 250]
    ] # [x,y]
    labels = [
        1, 
        0
    ] # 1: postiive. 0: negative
    
    # Predict
    _,_,painted_image = ctsam.first_frame_click(
        images[0],
        np.array(points), 
        np.array(labels), 
        multimask
    )
    path_out_img = os.path.join(path_out,"image")
    pathlib.Path(path_out_img).mkdir(exist_ok=True,parents=True)
    painted_image.image.save(os.path.join(path_out_img,"tmp_painted_img_{0:04d}.png".format(painted_image.idx)))
        
    masks, logits, painted_images = ctsam.generator(
        images,
    )
    
    # Save
    for mask, painted_image in zip(masks, painted_images):
        path_out_img = os.path.join(path_out,"image")
        pathlib.Path(path_out_img).mkdir(exist_ok=True,parents=True)
        painted_image.image.save(os.path.join(path_out_img,"tmp_painted_img_{0:04d}.png".format(painted_image.idx)))
        
        path_out_mask_ngp = os.path.join(path_out,"mask_instant-ngp")
        pathlib.Path(path_out_mask_ngp).mkdir(exist_ok=True,parents=True)
        name = os.path.join(path_out_mask_ngp,'dynamic_mask_{0:04d}.png'.format(img.id))
        cv2.imwrite(name,1-mask)

        
    # ### Main Loop
    # pbar = tqdm(img_ids)
    # for img_id in pbar:
    #     img = imgs[img_id]
    #     # Get tracked keypoints
    #     filter_idx = np.where(img.point3D_ids>=0)
    #     filtered_img = img.xys[filter_idx].round().astype(int)
    #     filtered_pts = img.point3D_ids[filter_idx]
        
    #     # Load image
    #     image_name = os.path.join(path_images,"{0:04d}.png".format(img.id))
    #     frame = cv2.imread(image_name, cv2.IMREAD_COLOR)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     # reset
    #     # samcontroler.sam_controler.reset_image()
    #     # samcontroler.sam_controler.set_image(frame)
        
    #     # Generate promts
    #     if img_id == img_ids[0]: 
    #         # First click
    #         points = [
    #             # [int(frame.shape[0]/2), int(frame.shape[1]/2)],
    #             [ 100, 500],
                
    #             [ 100, 250]
    #         ] # [x,y]
    #         labels = [
    #             1, 
    #             0
    #         ] # 1: postiive. 0: negative
            
    #         # Predict
    #         ctsam.first_frame_click(
    #             img_id,
    #             frame,
    #             np.array(points), 
    #             np.array(labels), 
    #             multimask
    #         )
            
    #         # samcontroler.sam_controler.reset_image()
    #         # samcontroler.sam_controler.set_image(frame)
    #         # mask, logit, painted_image = samcontroler.first_frame_click(
    #         #     frame, 
    #         #     np.array(points), 
    #         #     np.array(labels), multimask)
    #     else:
            
    #         non_zero_scores = np.array([score for score in pt_scores.values() if score != 0])
    #         score_std = np.std(np.abs(non_zero_scores))
    #         threshold = np.abs(non_zero_scores).mean()
        
    #         print('threshold: {} std: {}'.format(threshold,score_std))
    #         while True:
    #             points = []
    #             labels = []
                
    #             for pt_id, img_xy in zip(filtered_pts,filtered_img):
    #                 score = pt_scores[pt_id]
                    
    #                 #TODO: maybe only filter positive?
    #                 if score >=0:
    #                     if abs(score) < threshold: continue 
                    
    #                 label = 1 if score > 0 else -1
    #                 point = [img_xy[0],img_xy[1]]
                    
    #                 points.append(point)
    #                 labels.append(label)
    #             if len(points) == 0:
    #                 # reduce threshold
    #                 threshold -= score_std
    #                 print('reduce threshold to {}'.format(threshold))
    #             else:
    #                 break
        
    #         mask, logit, painted_image = samcontroler.first_frame_click(
    #             frame, 
    #             np.array(points), 
    #             np.array(labels), 
    #             multimask)
            
    #         # TRY mask input
    #         prompts = {
    #             'point_coords': np.array(points),
    #             'point_labels': np.array(labels),
    #             'mask_input': logit[None,:,:]
    #         }
    #         mask, logit, painted_image = samcontroler.predict(
    #             frame,
    #             prompts,
    #             multimask=True)
        
        
    #     # Update score
    #     mask_score = mask[filtered_img[:,1],filtered_img[:,0]] #
    #     for pt_id, score in zip(filtered_pts, mask_score):
    #         pt_scores[pt_id] += 1 if score == True else -1
        
    #     # Save
    #     path_out_img = os.path.join(path_out,"image")
    #     pathlib.Path(path_out_img).mkdir(exist_ok=True,parents=True)
    #     painted_image.save(os.path.join(path_out_img,"tmp_painted_img_{0:04d}.png".format(img.id)))
        
    #     path_out_mask_ngp = os.path.join(path_out,"mask_instant-ngp")
    #     pathlib.Path(path_out_mask_ngp).mkdir(exist_ok=True,parents=True)
    #     name = os.path.join(path_out_mask_ngp,'dynamic_mask_{0:04d}.png'.format(img.id))
    #     cv2.imwrite(name,1-mask)