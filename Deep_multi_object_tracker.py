__copyright__ = """
@copyright (c) 2024 by Robert Bosch GmbH. All rights reserved.

The reproduction, distribution and utilization of this file as 
well as the communication of its contents to others without express 
authorization is prohibited. Offenders will be held liable for the 
payment of damages and can be prosecuted. All rights reserved 
particularly in the event of the grant of a patent, utility 
model or design.
"""
from inputs.configs.model_config import *
from src.SiamNet import *
import torchvision.transforms.functional as F
import torch
import cv2
from  src.nms import NMS
from src import nms
from torch.autograd import Variable
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from src import seq_proc
from src.multi_object_tracker import Tracker
from os import path as osp
import json
class Detection(object):  #detection attributes initialization

    def __init__(self, tlwh, confidence, feature,fidx,template,disp,min_s_x,max_s_x,s_x,avg_chans):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.fidx=fidx
        self.template=template
        self.disp = disp
        self.min_s_x=min_s_x
        self.max_s_x = max_s_x
        self.s_x = s_x
        self.avg_chans = avg_chans

def geometrical_match(boxes_set1, boxes_set2):
    boxes_set1_tl, boxes_set1_br = boxes_set1[:2], boxes_set1[:2] + boxes_set1[2:]
    boxes_set2_tl, boxes_set2_br = boxes_set2[:, :2], boxes_set2[:, :2] + boxes_set2[:, 2:]

    topleft, botright = np.c_[np.maximum(boxes_set1_tl[0], boxes_set2_tl[:, 0])[:, np.newaxis],
    np.maximum(boxes_set1_tl[1], boxes_set2_tl[:, 1])[:, np.newaxis]], np.c_[
        np.minimum(boxes_set1_br[0], boxes_set2_br[:, 0])[:, np.newaxis],
        np.minimum(boxes_set1_br[1], boxes_set2_br[:, 1])[:, np.newaxis]]
    widht = np.maximum(0., botright - topleft)

    intersection_area = widht.prod(axis=1)
    boxes_set1_area, boxes_set2_area = boxes_set1[2:].prod(), boxes_set2[:, 2:].prod(axis=1)
    geometricalMatach = intersection_area / (boxes_set1_area + boxes_set2_area - intersection_area)
    return geometricalMatach

def extract_sequence_info(sequence,sequence_dir, detections):
   #cosolodating the attributes of the sequence such as image_filenames,detections,image_size,min_frame_idx,max_frame_idx into seq_info
   #sequence_dir : sequcen directry path
   #sequence: current seq name
   #detections: LH2 detections of the sequence

    image_dir = os.path.join(sequence_dir, sequence)
    image_filenames=[os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    image_filenames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    #THESE LINES TO TAKE EVRY 3RD EXPOSURE IN ALL THREE EXPOSERS
    #image_filenames_arr=np.asarray(image_filenames)
    #image_filenames1= image_filenames_arr[np.mod(np.arange(image_filenames_arr.size), 3) == 1]
    #image_filenames=image_filenames1.tolist()
    image = cv2.imread(image_filenames[0],cv2.IMREAD_GRAYSCALE)
    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    seq_info = {
        "image_filenames": image_filenames,
        "detections": detections,
        "image_size": image.shape,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info,image_dir

def featur_extractor_exempler(img,target_position,target_size):
    img_uint8 = cv2.imread(img)
    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_double = np.double(img_uint8)  # uint8 to float
    avg_chans = np.mean(img_double, axis=(0, 1))
    wc_z = target_size[1] + p.context_amount * sum(target_size)
    hc_z = target_size[0] + p.context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.examplar_size / s_z

    # crop examplar z in the first frame
    z_crop = SiamNet.get_subwindow_tracking(img_double, target_position, p.examplar_size, round(s_z), avg_chans)

    z_crop = np.uint8(z_crop)  # you need to convert it to uint8
    # convert image to tensor
    z_crop_tensor = 255.0 * F.to_tensor(z_crop).unsqueeze(0)
    d_search = (p.instance_size - p.examplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    # arbitrary scale saturation
    min_s_x = p.scale_min * s_x
    max_s_x = p.scale_max * s_x
    # extract feature for examplar z
    z_features = net.feat_extraction(Variable(z_crop_tensor).to(device))
    z_features = z_features.repeat(p.num_scale, 1, 1, 1)
    return z_features,min_s_x,max_s_x,s_x,avg_chans

def featur_extractor_searchregion(filename,target_position,target_size,s_x,avg_chans):
    img_uint8 = cv2.imread(filename)
    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_double = np.double(img_uint8)  # uint8 to float

    scaled_instance = s_x * scales
    scaled_target = np.zeros((2, scales.size), dtype=np.double)
    scaled_target[0, :] = target_size[0] * scales
    scaled_target[1, :] = target_size[1] * scales
    # extract scaled crops for search region x at previous target position
    x_crops = SiamNet.make_scale_pyramid(img_double, target_position, scaled_instance, p.instance_size, avg_chans, p)
    # get features of search regions
    x_crops_tensor = torch.FloatTensor(x_crops.shape[3], x_crops.shape[2], x_crops.shape[1], x_crops.shape[0])

    for k in range(x_crops.shape[3]):
        tmp_x_crop = x_crops[:, :, :, k]
        tmp_x_crop = np.uint8(tmp_x_crop)
        # numpy array to tensor
        x_crops_tensor[k, :, :, :] = 255.0 * F.to_tensor(tmp_x_crop).unsqueeze(0)

    # get features of search regions
    x_features = net.feat_extraction(Variable(x_crops_tensor).to(device))
    return x_features,scaled_instance,scaled_target

def run_tracker(p, net, target_position, target_size,filename,z_features,min_s_x,max_s_x,s_x,avg_chans):
        x_features,scaled_instance,scaled_target=featur_extractor_searchregion(filename, target_position, target_size,s_x,avg_chans)
        # evaluate the offline-trained network for exemplar x features
        target_position, new_scale,conf = SiamNet.tracker_eval(net, round(s_x), z_features, x_features, target_position, window, p)

        # scale damping and saturation
        s_x = max(min_s_x, min(max_s_x, (1 - p.scale_LR) * s_x + p.scale_LR * scaled_instance[int(new_scale)]))
        target_size = (1 - p.scale_LR) * target_size + p.scale_LR * np.array(
            [scaled_target[0, int(new_scale)], scaled_target[1, int(new_scale)]])

        # output bbox in the original frame coordinates
        o_target_position = target_position
        o_target_size = target_size
        center=np.array( [o_target_position[1], o_target_position[0],o_target_size[1],o_target_size[0]])
        bbox=np.array(
            [ o_target_position[1] - o_target_size[1] / 2,o_target_position[0] - o_target_size[0] / 2, o_target_size[1],
             o_target_size[0]])
        return bbox,center,z_features,s_x,conf

def detections_template(seq_info, frame_idx,  Components):
    # populating all the attributes of a bounding box such as width heigh,topleft, features, etc into detection_list_LH2
    #frame_idx: current frame index
    #Components: projection vectors to reduce the dimensiality of the feature vector

    frame_indices = seq_info["detections"][:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    detection_list_LH2 = []

    for row in seq_info["detections"][mask]:
        bbox = row[2:6]
        pos_x=row[2]+row[4]/2.0
        pos_y=row[3]+row[5]/2.0
        target_w=row[4]
        target_h=row[5]
        confidence = row[6]

        target_position = np.array([pos_y, pos_x], dtype=np.double)

        target_sz = np.array([target_h, target_w], dtype=np.double)
        #feature extraction of detections using Siamese network
        #templates_z_, min_s_x, max_s_x, s_x, avg_chans=featur_extractor_exempler(seq_info["image_filenames"][int(frame_idx / skip_frame)],
        #                                                                         target_position,target_sz)
        templates_z_, min_s_x, max_s_x, s_x, avg_chans = featur_extractor_exempler(
            seq_info["image_filenames"][int(frame_idx / skip_frame)-offset],
            target_position, target_sz) # if skipframe =1 andframe no starts with 1
        templates_z_1=templates_z_[0,:,:,:]
        feature = np.reshape(templates_z_1.cpu().detach().numpy(), templates_z_1.cpu().detach().numpy().size)
        #dimensionality of the feature is very high 30k approx an hence used PCA to reduce the dimensionality
        feature = feature.dot(Components)  #feaure size reduction
        detection_list_LH2.append(Detection(bbox, confidence, feature,frame_idx,templates_z_,(0,0), min_s_x,
                                            max_s_x, s_x, avg_chans))
    return detection_list_LH2

def process_next_frame_call(vis):  #main video processing pipelne for tracking

        if Backward_flag:  # flag for backward tracking: intitializing the backward track
            if vis.fl == 0:
                detections_LH2 = detections_template(seq_info, start_frame,Components)
                for i in range(0, len(detections_LH2)):
                    if (detections_LH2[i].tlwh[0] == vis.bbox[0]):
                        vis.prev_detections.append(
                        Detection(detections_LH2[i].tlwh, detections_LH2[i].confidence, detections_LH2[i].feature[0:n_comp],
                                  detections_LH2[i].fidx, detections_LH2[i].template, detections_LH2[i].disp,detections_LH2[i].min_s_x,detections_LH2[i].max_s_x, detections_LH2[i].s_x, detections_LH2[i].avg_chans))

                vis.fl = 1
            vis.frame_idx = max(0, vis.start_frame1 - skip_frame)
            vis.start_frame1 = vis.frame_idx

        #image = cv2.imread(seq_info["image_filenames"][int(vis.frame_idx / skip_frame)-offset], cv2.IMREAD_COLOR)
        print(len(seq_info['image_filenames']))
        print("vis.frame_idx = ",vis.frame_idx )
        print("skip_frame = ",skip_frame)
        print("offset = ",offset)
        print("int(vis.frame_idx / skip_frame)-offset = ", int(vis.frame_idx / skip_frame)-offset)
        detections_LH2= detections_template(seq_info, vis.frame_idx, Components) # populating all the attributes of a bounding box such as width heigh,topleft, features, etc into detection_LH2

        detections_Superset= detections_LH2= [d for d in detections_LH2 if d.confidence >= vis.min_confidence] #Selecting the detections with high confidance
         #####block that appends the existing LH2 detections with high confidant siamese predictions, resulting detections_Superset
        if Backward_flag:
            detections_Superset=[]
        detections_Superset, bboxes,vis.mx = siamese_prediction(seq_info, detections_Superset, vis)
        print(vis.frame_idx)
        if ((Backward_flag) & (
                bboxes != [])):  ##appending the LH2 detection in case there is a overlap between LH2 det and siamese preditions
            bboxe = [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]
            bboxe = np.asarray(bboxe)
            for ii in range(0, detections_LH2.__len__()):
                reshaped_detection = np.reshape(
                    [detections_LH2[ii].tlwh[0], detections_LH2[ii].tlwh[1], detections_LH2[ii].tlwh[2],
                     detections_LH2[ii].tlwh[3]], (1, 4))
                match = geometrical_match(bboxe, reshaped_detection)
                if (match > 0.8):
                    detections_Superset.append(detections_LH2[ii])

        boxes = np.array([d.tlwh for d in detections_Superset])
        scores = np.array([d.confidence for d in detections_Superset])
        indices,matches = nms.nms_algorithm(
            boxes, vis.nms_max_overlap, scores)

        for i in range(matches.__len__()): #preserving the flow vector
            detections_Superset[matches[i][0]].disp=detections_Superset[matches[i][1]].disp

        prev_detections=detections_Superset = [detections_Superset[i] for i in indices]

        for track in tracker.tracks:     ##update track age
            track.age += 1
            track.last_update += 1
        #match detections (detections_Superset) with trackes and update tracks with detections based on match
        tracker.state_update(detections_Superset,Backward_flag,ID,newtrack_th,model_th,vis.min_confidence,seq_info["image_size"])

        for track in tracker.tracks:
            bbox = track.mean[:4].copy()
            bbox[2] *= bbox[3]
            bbox[:2] -= bbox[2:] / 2

            if (not (track.state == 2)) or (track.last_update > 0):
                continue
            results.append([
                vis.frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.confidence])
        return  prev_detections,vis.frame_idx,vis.fl,boxes,vis.start_frame1,vis.mx


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", help="modelpath",
        default=None, required=True)
    return parser.parse_args()

def siamese_prediction(seq_info,detections_Superset,vis):
     #boundary = []
     bboxes=[]
     #boundary.append(seq_info["image_size"][0] * 0.25) # this is constant no need to change
     #boundary.append(seq_info["image_size"][1] * 0.25) # this is constant no need to change

     if len(vis.prev_detections) != 0:
        # predicting the location of each previous detection in the current frame using Siamese
        for i1 in range(0, len(vis.prev_detections)):

            # reading previous frame bounding box co-ordinates and its template
            pos_x = vis.prev_detections[i1].tlwh[0] + vis.prev_detections[i1].tlwh[2] / 2.0
            pos_y = vis.prev_detections[i1].tlwh[1] + vis.prev_detections[i1].tlwh[3] / 2.0
            target_w = vis.prev_detections[i1].tlwh[2]
            target_h = vis.prev_detections[i1].tlwh[3]
            prev_template = vis.prev_detections[i1].template
            # reading current filename/frame
            #filename = seq_info["image_filenames"][int(vis.frame_idx / skip_frame)]
            filename = seq_info["image_filenames"][int(vis.frame_idx / skip_frame)-offset]

            target_position = np.array([pos_y, pos_x], dtype=np.double)
            #bbox = np.array([pos_x - vis.prev_detections[i1].disp[0], pos_y - vis.prev_detections[i1].disp[1]])
            #target_position = np.array([pos_y- vis.prev_detections[i1].disp[1], pos_x- vis.prev_detections[i1].disp[0]], dtype=np.double)
            target_sz = np.array([target_h, target_w], dtype=np.double)
            min_s_x=vis.prev_detections[i1].min_s_x
            max_s_x = vis.prev_detections[i1].max_s_x
            s_x = vis.prev_detections[i1].s_x
            avg_chans = vis.prev_detections[i1].avg_chans

            bboxes, center, templates_z_, s_x,conf = run_tracker(p, net,target_position,target_sz,filename,
                                                             prev_template,min_s_x,max_s_x,s_x,avg_chans)

            # Matching of 'prev_template' of in the neighborhood of location 'bbox' in the current image 'filename'
            # bboxes is the bounding box of matched location  and matched with confidance conf and its template templates_z_

            ##checking if the predicted bounding box flow vector direction does not deviate heavility from the previous detection's
            # flow vector with flag fll. fll=1 means predicted boundnding box is spurious and should be ignored
            if(conf>vis.mx):
                vis.mx=conf
                ##added by chiru on 2025 March 19
            prev_bbox_cen = np.array([pos_x, pos_y])
            motion_vect = (prev_bbox_cen[0] - center[0]), (prev_bbox_cen[1] - center[1])
            disp = motion_vect
            mag = np.sqrt(motion_vect[0] * motion_vect[0] + motion_vect[1] * motion_vect[1])
            prev_mv_mag = np.sqrt(
                    vis.prev_detections[i1].disp[0] * vis.prev_detections[i1].disp[0] + vis.prev_detections[i1].disp[
                        1] * vis.prev_detections[i1].disp[1])
            motion_similarity = np.minimum(mag, prev_mv_mag) / np.maximum(mag, prev_mv_mag)
            motion_dir_similarity = ((np.sign(disp[0]) == np.sign(vis.prev_detections[i1].disp[0])) & (
                        (np.sign(disp[1])) == np.sign(vis.prev_detections[i1].disp[1])))
            motionflag = 0
            mflag = 0
            if (vis.prev_detections[i1].disp[0] == 0) | (vis.prev_detections[i1].disp[1] == 0):
                    mflag = 1
            if ((motion_dir_similarity) | (motion_similarity > 0.7)) | (mflag == 1):
                    motionflag = 1


            #fll = 0  # flag to detect spurios predictions
            #if ((vis.prev_detections[i1].disp[0] != 0) & (vis.prev_detections[i1].disp[1] != 0)):
            #    if ((np.sign(disp[0]) != np.sign(vis.prev_detections[i1].disp[0])) | (
            #                (np.sign(disp[1])) != np.sign(vis.prev_detections[i1].disp[1]))):
                    # checking if the direction of current and previous frame flow vectors of a particular detection is different or not

             #       val = abs(disp[0]) + abs(
             #           disp[1])  # if the directions are diff then check the magnitude of current flow
             #       if val > 50:
             #           fll = 1

            # Block that creates superset over existing LH2 detection--detections_Superset

            if ((~vis.Backward_flag)&(conf >= vis.append_th) & (motionflag) & (1)):
                detections_Superset.append(
                    Detection(bboxes[0:4], conf, vis.prev_detections[i1].feature, vis.frame_idx, templates_z_,
                              disp,vis.prev_detections[i1].min_s_x,vis.prev_detections[i1].max_s_x, s_x, vis.prev_detections[i1].avg_chans))  ###consider to change detection variable
            if (Backward_flag & (conf >= 0.04) & (bboxes[2] >= 16) & (bboxes[3] >= 16)):
                detections_Superset.append(
                    Detection(bboxes[0:4], conf, vis.prev_detections[i1].feature, vis.frame_idx, templates_z_,
                              disp, vis.prev_detections[i1].min_s_x, vis.prev_detections[i1].max_s_x, s_x,
                              vis.prev_detections[i1].avg_chans))

     return  detections_Superset,bboxes,vis.mx

def plottrack3d(numFrames,skip_frame,frame_name_list,detections_GT_file,output,video,video_folder,plot3d):
    font = cv2.FONT_HERSHEY_SIMPLEX
    detections_GT = np.loadtxt(detections_GT_file, delimiter=',')
    detections_GT.astype(int)
   # uniqueIDs_GT = np.unique(detections_GT[:, 0])
    car_color=(0, 255, 0)
    truck_color=(0, 0, 255)
    objID_color=(255, 255, 255)
    color_thickness=2
    for k in range(numFrames):
        frame_name=frame_name_list[int(k / skip_frame)]
        a = np.array(np.where(np.equal((detections_GT[:, 0]), k)))
        a = a.flatten()
        if ((a.size != 0)):
            img = cv2.imread(video_folder+ "/" + video+ "/"  + frame_name)
        if a.any():
            for j in range(0, len(a)):
                bbox = detections_GT[a[j], 2:6]
                bbox1 = detections_GT[a[j], 8:12]
                obj_class = detections_GT[a[j], 7]
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                p11 = (int(bbox[0] + bbox[2]), int(bbox[1]))
                p22 = (int(bbox[0]), int(bbox[1] + bbox[3]))
                p3 = (int(bbox1[0]), int(bbox1[1]))
                p4 = (int(bbox1[2]), int(bbox1[3]))

                w2 = (bbox1[0] - (bbox[0] + bbox[2]))
                h2 = bbox1[3] - bbox[1]
                if obj_class == 1:
                    if w2 > 0:

                        cv2.rectangle(img, p1, p2, car_color, color_thickness, 1)
                        cv2.line(img, p11, p3, car_color, color_thickness, 1)
                        cv2.line(img, p3, p4, car_color, color_thickness, 1)
                        cv2.line(img, p2, p4, car_color, color_thickness, 1)

                    else:

                        cv2.rectangle(img, p1, p2, car_color, color_thickness, 1)
                        cv2.line(img, p1, p3, car_color, color_thickness, 1)
                        cv2.line(img, p3, p4, car_color, color_thickness, 1)
                        cv2.line(img, p22, p4, car_color, color_thickness, 1)
                if obj_class == 2:
                    if w2 > 0:

                        cv2.rectangle(img, p1, p2, truck_color, color_thickness, 1)
                        cv2.line(img, p11, p3, truck_color, color_thickness, 1)
                        cv2.line(img, p3, p4, truck_color, color_thickness, 1)
                        cv2.line(img, p2, p4, truck_color, color_thickness, 1)

                    else:

                        cv2.rectangle(img, p1, p2, truck_color, color_thickness, 1)
                        cv2.line(img, p1, p3, truck_color, color_thickness, 1)
                        cv2.line(img, p3, p4, truck_color, color_thickness, 1)
                        cv2.line(img, p22, p4, truck_color, color_thickness, 1)

                id = int(detections_GT[a[j], 1])
                if id != 0:
                    cv2.putText(img, str(id), p3, font, 0.8, objID_color, color_thickness)
        if plot3d:
         if ((a.size != 0)):
            cv2.imwrite(output + frame_name, img)


def iou_evaluation_IDinterpolate(detections, tracks,detections_mod,track_mod):  # tracks gt detections is tracks
    detections1 = detections
    detection_indices = list(range(len(detections)))
    track_indices = list(range(len(tracks)))
    cost_matrix = np.zeros((len(detection_indices), len(track_indices)))
    for row, det_idx in enumerate(detection_indices):

        if len(track_indices) == 0:
            detections1[row] = np.append(0, detections1[row])
        else:

            bbox = detections[det_idx]

            candidates = np.array(tracks)

            bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
            candidates_tl = candidates[:, :2]
            candidates_br = candidates[:, :2] + candidates[:, 2:4]

            tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                       np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
            br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                       np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
            wh = np.maximum(0., br - tl)

            area_intersection = wh.prod(axis=1)
            area_bbox = bbox[2:].prod()
            area_candidates = candidates[:, 2:4].prod(axis=1)

            overlap = area_intersection / (area_bbox + area_candidates - area_intersection)
            cost_matrix[row, :] = overlap
            fl = 0
            for j in range(len(track_indices)):
                if cost_matrix[row, j] > 0.5:
                    fl = 1
                    detections1[row] = np.append(candidates[j][4], detections1[row])
                    #track_mod[j][10:14] = detections_mod[row][8:12] ##uncomment hithis if required
            if fl == 0:
                detections1[row] = np.append(0, detections1[row])
            track_mod=detections1
    return detections1,track_mod


def sequence_info(detections, gt):
    seq_info = {
        "detections": detections,
        "track": gt,
    }
    return seq_info


def id_interpolate_algo(seq_info, frame_idx):
    frame_indices = seq_info["detections"][:, 0].astype(np.int)
    gt_indices = seq_info["track"][:, 0].astype(np.int)

    mask = frame_indices == frame_idx
    maskgt = gt_indices == frame_idx

    track_can = []
    detections_can = []
    detections_mod = []
    a = [0, 0, 0, 0]
    track_mod = []
    for row in seq_info["detections"][mask]:
        bbox = row[2:6]
        detections_can.append(bbox)
        detections_mod.append(row)

    for row in seq_info["track"][maskgt]:
        kk = []
        bbox = np.append(row[2:6], row[1])
        track_can.append(bbox)
        kk.append(row)
        track_mod1 = np.append(kk, a[0:4])
        track_mod.append(track_mod1)
    detections1,track_mod = iou_evaluation_IDinterpolate(detections_can, track_can,detections_mod,track_mod)
    for i in range(0, len(detections1)):
        detections_mod[i][1] = detections1[i][0]
    return detections_mod,track_mod
def trackID_interpolation(detection_file,track_file,detection_file_mod,detection_corrected,track_mod):
    ff = open(detection_file_mod, 'w')
    ff1 = open(detection_corrected, 'w')
    ff2 = open(track_mod, 'w')
    detections_LH2 = np.loadtxt(detection_file)
    #detections_LH2 = np.loadtxt(detection_file, delimiter=',')
    detections_LH2 = np.asarray(detections_LH2)
    gt = np.loadtxt(track_file, delimiter=',')
    gt = np.asarray(gt)


    a = detections_LH2[:, 0].astype(np.int)
    list_1 = a.tolist()
    frame_indices = np.array(list_1)
    seq_info = sequence_info(detections_LH2, gt)
    frame_indices1 = np.unique(frame_indices)
    for i in range(0, frame_indices1.shape[0]):

        detections_mod,track_mod = id_interpolate_algo(seq_info, frame_indices1[i])

        for row in range(0, len(detections_mod)):
            print('%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d' % (
            int(detections_mod[row][0]), int(detections_mod[row][1]), int(detections_mod[row][2]),
            int(detections_mod[row][3]), int(detections_mod[row][4]), int(detections_mod[row][5]),
            detections_mod[row][6], detections_mod[row][7],
            int(detections_mod[row][8]), int(detections_mod[row][9]), int(detections_mod[row][10]),
            int(detections_mod[row][11])), file=ff)

        for row in range(0, len(track_mod)):
            print('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (
                int(track_mod[row][0]), int(track_mod[row][1]), int(track_mod[row][2]),
                int(track_mod[row][3]), int(track_mod[row][4]), int(track_mod[row][5]), track_mod[row][6],
                track_mod[row][7],
                int(track_mod[row][8]), int(track_mod[row][9]), int(track_mod[row][10]),
                int(track_mod[row][11]), int(track_mod[row][12]), int(track_mod[row][13])), file=ff2)


    detections_LH2 = np.loadtxt(detection_file_mod, delimiter=',')

    k1 = [k[1] for k in detections_LH2]
    MAX_ID = max(k1)
    hist = []
    row = int(MAX_ID + 1)
    col = 3
    hist = np.full((row, col), 0)

    for i in range(0, detections_LH2.shape[0]):
        hist[int(detections_LH2[i][1])][int(detections_LH2[i][7])] = hist[int(detections_LH2[i][1])][
                                                                         int(detections_LH2[i][7])] + 1

    final_cl = []

    for i in range(0, int(MAX_ID + 1)):
        if hist[i][1] > hist[i][2]:
            final_cl.append(1)
        else:
            final_cl.append(2)
    final_cl = np.array(final_cl)
    for row in range(0, detections_LH2.shape[0]):
        detections_LH2[row][7] = final_cl[int(detections_LH2[row][1])]
        print('%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d' % (
        int(detections_LH2[row][0]), int(detections_LH2[row][1]), int(detections_LH2[row][2]),
        int(detections_LH2[row][3]), int(detections_LH2[row][4]), int(detections_LH2[row][5]), detections_LH2[row][6],
        detections_LH2[row][7],
        int(detections_LH2[row][8]), int(detections_LH2[row][9]), int(detections_LH2[row][10]),
        int(detections_LH2[row][11])), file=ff1)

    ff.close()
def load_cfgs(checkpoint):
  if osp.isdir(checkpoint):
    train_dir = checkpoint
  else:
    train_dir = osp.dirname(checkpoint)

  with open(osp.join(train_dir, 'params.json'), 'r') as f:
    params = json.load(f)
  return params
def find_FB_candidate_boxes(F_tracks_file):

    detections_ID = np.loadtxt(F_tracks_file, delimiter=',')
    uniqueIDs = np.unique(detections_ID[:, 1])
    numTracks = len(uniqueIDs)
    candidatesFBTrackIdx = np.zeros((numTracks, 6))
    count_FBTrackIdx = 0
    for trackID in range(0, numTracks):
      a = np.array(np.where(np.equal(detections_ID[:, 1], uniqueIDs[trackID])))
      a = a.flatten()
      temp = detections_ID[:, 5]
      temp_frames = detections_ID[:, 0]
      temp_frames_1 = temp_frames[a]
      objHeight = temp[a]
      k1 = len(temp_frames_1)
      objHeightRatio = objHeight[-1] / objHeight[0]
      if ((objHeightRatio > 1.2)|(detections_ID[a[0],6]>0.7)):
      #if (objHeightRatio > 1.1):
        idx = a[0]
        candidatesFBTrackIdx[count_FBTrackIdx, :] = detections_ID[idx, 0:6]
        count_FBTrackIdx = count_FBTrackIdx + 1
    return candidatesFBTrackIdx,count_FBTrackIdx
def write_filtered_outputs(out,out_F,out_B):

    df_B=np.loadtxt(out_B, delimiter=',')
    df_F =np.loadtxt(out_F, delimiter=',')
    if df_B.size!=0:
         df_FB = np.concatenate((df_B,df_F),axis=0)
    else:
     df_FB=df_F
    ff = open('raw.txt', 'w')
    for row in df_FB:
        ff.write(
            '%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n' % (row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5])))
    ff.close()
    nms = NMS()
    nms.convertor('raw.txt', out)

if __name__ == "__main__":
    # parameters to change across dataset: seq folder, skip_frame, file list, det files, thresholdconf,for kitti self.min_confidence=50, include overlap corr
    #args = parse_args()
    p = Config()
    #print(torch.__version__)   ##--checkpoint=./models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load(p.net)
    net = net.to(device)
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size * p.response_UP), np.hanning(p.score_size * p.response_UP))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size * p.response_UP, p.score_size * p.response_UP))
    window = window / sum(sum(window))

    # pyramid scale search
    scales = p.scale_step ** np.linspace(-np.ceil(p.num_scale / 2), np.ceil(p.num_scale / 2), p.num_scale)
    # evaluation mode
    net.eval()
    # choose which model to run
    #checkpoint=args.checkpoint
    checkpoint ='./models'

    Components = np.load('pcailc_vgg16_prewts_62.npy') ##BETTER
    params = load_cfgs(checkpoint)
    sequence_dir = params['sequence_dir']
    skip_frame = params['skip_frame']
    n_comp = params['n_comp']
    max_cosine_distance=params['max_cosine_distance']
    nn_budget=params['nn_budget']
    max_iou_distance = params['max_iou_distance']
    max_age = params['max_age']
    n_init = params['n_init']
    nms_max_overlap = params['nms_max_overlap']
    min_confidence = params['min_confidence']
    plot=params['plot']
    mod_det=params['mod_det']
    mod_track=params['mod_track']
    corrected_det = params['corrected_det']
    visualizations=params['visualizations']
    det_file=params['det_file']
    track_file=params['track_file']
    visualizations_3d=params['visualizations_3d']
    plot3d=params['plot3d']
    input_seq=params['input_seq']
    append_th=params['append_th']
    newtrack_th = params['newtrack_th']
    model_th=params['model_th']

    f = open(input_seq, 'r')
    x = f.read().splitlines()
    Components = Components[:, 0:n_comp]

    for sequence in x:
      mx=0
      print(sequence)
      out_dir = visualizations+sequence+'/'
      output_file=track_file+sequence+'.txt'
      detection_file = det_file +  sequence + '.txt'
      output_file = './tmp/' + sequence + '.txt'
      output_file_FB = './tmp/' + sequence + '_FB' + '.txt'
      out_final = out_dir + sequence + '.txt'
      if not os.path.exists(out_dir):
          os.makedirs(out_dir)

      if not os.path.exists(track_file):
          os.makedirs(track_file)

      detections_LH2 = np.loadtxt(detection_file, delimiter=',')
      #detections_LH2 = np.loadtxt(detection_file)
      detections_LH2 = np.asarray(detections_LH2)
      #reads LH2 detections of a sequence and extract sequence information
      seq_info, image_dir = extract_sequence_info(sequence, sequence_dir, detections_LH2)
      if (skip_frame==1):
          offset=seq_info['min_frame_idx']
      else:
          offset=0
      results = []
      bbox = []
      Backward_flag = False
      start_frame = 0
      ID = 1000

      tracker = Tracker(max_cosine_distance,max_iou_distance,max_age,n_init, nn_budget)
          # intializing the object
      tracker_wrapper = seq_proc.tracker_wrapper(seq_info,
        start_frame,skip_frame,Backward_flag,n_comp, bbox,ID,   Components,
        out_dir, nms_max_overlap,min_confidence,plot,window,scales,offset,mx,append_th,newtrack_th,model_th)
          #processing sequence recursively to track the object
      tracker_wrapper.run(process_next_frame_call)
      ff = open(output_file, 'w')
      for row in results:
              print(
                  '%d,%d,%d,%d,%d,%d,%f,-1,-1,-1' % (row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5]),row[6]),
                  file=ff)
      ff.close()
      ###  added by chiru on mar,24
      Backward_flag = False

      if Backward_flag:  # Backward tracking
          candidatesFBTrackIdx, count_FBTrackIdx = find_FB_candidate_boxes(output_file)

          results = []
          for idx in range(0, count_FBTrackIdx):
              tracker = Tracker(max_cosine_distance,max_iou_distance,max_age,n_init, nn_budget)
              bbox = candidatesFBTrackIdx[idx, 2:6]
              ID = candidatesFBTrackIdx[idx, 1]
              start_frame = int(candidatesFBTrackIdx[idx, 0])
              deep_tracker_wrapper = seq_proc.tracker_wrapper(seq_info, start_frame, skip_frame, Backward_flag,
                                     n_comp, bbox, ID, Components, out_dir, nms_max_overlap,min_confidence,plot,
                                                                   window,scales,offset,mx,append_th,newtrack_th,model_th)
              deep_tracker_wrapper.run(process_next_frame_call)

          ff = open(output_file_FB, 'w')
          for row in results:
              print('%d,%d,%d,%d,%d,%d,1,-1,-1,-1' % (
                  row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5])), file=ff)
          ff.close()
          write_filtered_outputs(out_final, output_file, output_file_FB)
      #detection_file_mod = mod_det + sequence + ".txt"
      #detection_corrected = corrected_det + sequence + ".txt"
      #detection_file_mod = mod_det + "modified_det" + ".txt"
      #detection_corrected = corrected_det + "corrected_det" + ".txt"
      #if not os.path.exists(mod_track):
      #    os.makedirs(mod_track)
      #track_mod = mod_track + sequence + ".txt"
      #trackID_interpolation(detection_file,output_file,detection_file_mod,detection_corrected,track_mod)
      #output = visualizations_3d + "/" + sequence + "/"
      #if not os.path.exists(output):
      #    os.makedirs(output)
      #frame_name_list = [f for f in os.listdir(image_dir) if f.endswith(".png")]
      #frame_name_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
      #plottrack3d(seq_info['max_frame_idx'],skip_frame,frame_name_list,detection_corrected,output,sequence,args.sequence_dir,plot3d)



