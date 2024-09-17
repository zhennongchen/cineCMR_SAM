# import os
# import numpy as np
# import cv2

# import torch

# from utils.util import save_png, mkdirs

# from utils.infer_utils import draw_contour, draw_contour_2, draw_contour_prediction, draw_contour_prediction_line

# import cv2
# # from skan import skeleton_to_csgraph
# # from skimage import morphology
# import math
# from PIL import Image
# from torchvision.transforms import functional as F


# def save_as_png(img, label, pred, cfg, file_name, dest_dir = None, test_type = None):
#     sh = np.shape(img)
#     fr = sh[0]
    
#     dest_dir = os.path.join(dest_dir, test_type, "D_RATIO_" + str(cfg.trainer.Usage_percent))
    
#     if not os.path.exists(dest_dir):
#         mkdirs([dest_dir])

#     for dcm_frame_name in range(fr):
#         folder_path = os.path.join(dest_dir, file_name)
#         pred_path = os.path.join(dest_dir, file_name)
        
#         if not os.path.exists(folder_path):
#             mkdirs([folder_path])
#             mkdirs([pred_path])
            
        
    
#         # input_image = (Denormalization(input_batch["input"][0, 0, dcm_frame_name, :, :].cpu().numpy())).astype(dtype=np.uint8)
#         if isinstance(img, np.ndarray):
#             input_image = img[dcm_frame_name, 0, ...]
#         else:
#             input_image = img[dcm_frame_name, 0, ...].cpu().numpy()
#         # *255
#         input_image = input_image.astype(dtype=np.uint8)
#         input_image = np.expand_dims(input_image, axis=2)
        
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#         mask = pred[dcm_frame_name,0,...].cpu().numpy()
#         fr_label = label[0, dcm_frame_name,...]

#         input_image_with_contour, flag_annotaed_fr = draw_contour(
#             input_image.copy(), fr_label , mask, num_classes=4
#         )
#         if flag_annotaed_fr == True:
#             result_dir = os.path.join(pred_path, str(dcm_frame_name) + "_A")
#         else:
#             result_dir = os.path.join(pred_path, str(dcm_frame_name))
#         save_png(result_dir + ".png", input_image_with_contour)
#     return 

# # def calc_lvef(A2CH_file_name, A4CH_file_name, a2c_net_list, a4c_net_list , opt = False):
# #     measure_handler = ClinicalMeasurement(a2c_x, a2c_y, a4c_x, a4c_y)

# #     ED_LA_long_axis, ED_LV_long_axis = measure_handler.calc_long_axis(separated_a2c_EDES_dict["ED"],
# #                                                                       separated_a4c_EDES_dict["ED"])
# #     ES_LA_long_axis, ES_LV_long_axis = measure_handler.calc_long_axis(separated_a2c_EDES_dict["ES"],
# #                                                                       separated_a4c_EDES_dict["ES"])

# #     ED_LV_volume = measure_handler.calc_volume(
# #         separated_a2c_EDES_dict["ED"]["LV cavity"], separated_a4c_EDES_dict["ED"]["LV cavity"], ED_LV_long_axis
# #     )
# #     ES_LV_volume = measure_handler.calc_volume(
# #         separated_a2c_EDES_dict["ES"]["LV cavity"], separated_a4c_EDES_dict["ES"]["LV cavity"], ES_LV_long_axis
# #     )

# #     stroke_volume = ED_LV_volume - ES_LV_volume

# #     EF = stroke_volume / ED_LV_volume * 100

# #     final_json = {"EF": round(EF,1),
# #                   "LVEDV": round(ED_LV_volume,1),
# #                   "LVESV": round(ES_LV_volume,1),
# #                   "files": [
# #                         {"filename": A2CH_file_name,
# #                         "view": "A2C"},
# #                         {"filename": A4CH_file_name,
# #                         "view": "A4C"}]}

# #     view_mask_list = [a2c_predicted_masks, a4c_predicted_masks]
# #     ED_ES_dict_list = [a2c_EDES_dict, a4c_EDES_dict]
# #     class_list = [cfg.data.list.A2CH, cfg.data.list.A4CH]

# #     for idx, predicted_masks in enumerate(view_mask_list):
# #         test = []
# #         for frame, masks_per_frame in enumerate(predicted_masks):
# #             contour = make_contour(masks_per_frame, class_list[idx], 30, False)
# #             if frame == (ED_ES_dict_list[idx]['ES_frame']):
# #                 item = {"frame_idx": frame, "types": "Contour", "phase": "ES", "labels": contour}
# #             elif frame == (ED_ES_dict_list[idx]['ED_frame']):
# #                 item = {"frame_idx": frame, "types": "Contour", "phase": "ED", "labels": contour}
# #             else:
# #                 item = {"frame_idx": frame, "types": "Contour", "phase": "","labels": contour}
# #             test.append(item)
# #         final_json["files"][idx]["frames"] = test

# #     # json.dump(
# #     #     final_json,
# #     #     open("api_lvef/sample_data/results.json", "w"),
# #     #     cls=NumpyEncoder,
# #     #     indent=4,
# #     #     sort_keys=False,
# #     # )
# #     # print(final_json)
# #     # return final_json

# #     return json.dumps(final_json, cls=NumpyEncoder, indent=4, sort_keys=False,)


# # class Line_measurement:
# #     def __init__(self, predicted_mask, class_list):
# #         self.predicted_mask = predicted_mask
# #         self.class_list = class_list
# #         self.pts = self.mask_to_pts(self.predicted_mask)
# #         self.skel_pts = {"Aorta": [], "LA": [], "IVS": [], "LV posterior wall": [], "mv anterior leaflet": [],
# #                     "mv posterior leaflet": []}

# #     def mask_to_pts(self, predicted_mask):
# #         for idx, class_label in enumerate(self.class_list):
# #             image, contours_pred, _ = cv2.findContours(
# #                 (predicted_mask == idx + 1).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
# #             )

# #             # choose largest component
# #             shape = np.shape(contours_pred)

# #             if shape[0] > 1:
# #                 pt_max = 0
# #                 for i in range(shape[0]):
# #                     if pt_max < np.shape(contours_pred[i])[0]:
# #                         pt_max = np.shape(contours_pred[i])[0]
# #                         out = np.reshape(contours_pred[i], (pt_max, 2))
# #             else:
# #                 out = np.reshape(contours_pred[i], (shape[1], 2))
# #         return out

# #     def skeletonize_pts(self, img, pts, color=(255, 255, 255), flag=False):
# #         mask = np.zeros_like(img)
# #         mask_sep = cv2.drawContours(mask, [pts], -1, color, -1)
# #         skeleton0 = morphology.skeletonize(mask_sep)
# #         pixel_graph, coordinates = skeleton_to_csgraph(skeleton0)
# #         skeletonized_pts = np.int64(coordinates[1:, 0:2])
# #         if flag == False:
# #             return skeletonized_pts
# #         elif flag == True:
# #             for item in skeletonized_pts:
# #                 cv2.drawMarker(img, (item[1], item[0]), (255, 0, 255), markerType=cv2.MARKER_STAR,
# #                                markerSize=3, thickness=2, line_type=cv2.LINE_AA)
# #             return skeletonized_pts, img

# #     def slope(self, pt1, pt2):
# #         x1 = pt1[0]
# #         y1 = pt1[1]
# #         x2 = pt2[0]
# #         y2 = pt2[1]
# #         if x2 != x1:
# #             return ((y2 - y1) / (x2 - x1))
# #         else:
# #             return 'NA'

# #     def draw_pts(self, img, pts , color = (255, 255, 255)):
# #         cv2.drawMarker(img, (pts[1], pts[0]),
# #                        color, markerType=cv2.MARKER_STAR,
# #                        markerSize=3, thickness=2, line_type=cv2.LINE_AA)
# #         return img

# #     def drawLine(self, image, pt1, pt2):
# #         x1 = pt1[0]
# #         y1 = pt1[1]
# #         x2 = pt2[0]
# #         y2 = pt2[1]
# #         m = self.slope(pt1, pt2)
# #         h, w = image.shape[:2]
# #         if m != 'NA':
# #             ### here we are essentially extending the line to x=0 and x=width
# #             ### and calculating the y associated with it
# #             ##starting point
# #             px = 0
# #             py = -(x1 - 0) * m + y1
# #             ##ending point
# #             qx = w
# #             qy = -(x2 - w) * m + y2
# #         else:
# #             ### if slope is zero, draw a line with x=x1 and y=0 and y=height
# #             px, py = x1, 0
# #             qx, qy = x1, h
# #         cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 255), 1)

# #     def drawLine_with_slope(self, image, pt, angle):
# #         angle = math.degrees(np.arctan(angle))
# #         x = pt[0]
# #         y = pt[1]
# #         imheight, imwidth = image.shape[:2]
# #         x1_length = (x - imwidth) / math.cos(angle)
# #         y1_length = (y - imheight) / math.sin(angle)
# #         length = max(abs(x1_length), abs(y1_length))
# #         endx1 = x + length * math.cos(math.radians(angle))
# #         endy1 = y + length * math.sin(math.radians(angle))

# #         x2_length = (x - imwidth) / math.cos(angle + 180)
# #         y2_length = (y - imheight) / math.sin(angle + 180)
# #         length = max(abs(x2_length), abs(y2_length))
# #         endx2 = x + length * math.cos(math.radians(angle + 180))
# #         endy2 = y + length * math.sin(math.radians(angle + 180))
# #         cv2.line(image, (int(endx1), int(endy1)), (int(endx2), int(endy2)), (255, 255, 255), 1)

# #     def find_tip_pts(self, pts):
# #         min = 512
# #         min_idx = 0
# #         for idx, item in enumerate(pts):  # 0(x)
# #             if min > item[0]:
# #                 min_idx = idx
# #                 min = item[0]
# #         tip_pts = pts[min_idx]
# #         return tip_pts

# #     def find_closest_pts_in_cts(slef, pts, cts):
# #         min = 512
# #         for idx, i in enumerate(cts):
# #             sqrt = np.sqrt((i[0] - pts[0])* (i[0] - pts[0]) + (i[1] - pts[1])* (i[1] - pts[1]))
# #             if min > sqrt:
# #                 min = sqrt
# #                 min_idx = idx
# #         closest_IVS_pts = cts[min_idx]
# #         return closest_IVS_pts

# #     def find_closest_pts_btw_cts(self, cts1, cts2):
# #         min = 512
# #         min_idx =0
# #         min_idx2 =0
# #         for idx, i in enumerate(cts1):
# #             for idx2, i2 in enumerate(cts2):
# #                 sqrt = np.sqrt((i[0] - i2[0])* (i[0] - i2[0]) + (i[1] - i2[1])* (i[1] - i2[1]))
# #                 if min > sqrt:
# #                     min = sqrt
# #                     min_idx = idx
# #                     min_idx2 = idx2
# #         closest_pts1= cts1[min_idx]
# #         closest_pts2 = cts2[min_idx2]
# #         return closest_pts1, closest_pts2

# #     def find_tip_pts_btw_cts(self, cts1, cts2):
# #         min = 2
# #         min_idx = 0
# #         min_idx2 = 0
# #         ct1_pts = []
# #         ct2_pts = []
# #         for idx, i in enumerate(cts1):
# #             for idx2, i2 in enumerate(cts2):
# #                 sqrt = np.sqrt((i[0] - i2[0]) * (i[0] - i2[0]) + (i[1] - i2[1]) * (i[1] - i2[1]))
# #                 if sqrt < min:
# #                     ct1_pts.append(i2)
# #                     ct2_pts.append(i)

# #         # find min , max from i
# #         tip_pts1, tip_pts2 = ct2_pts[0], ct2_pts[-1]
# #         return tip_pts1, tip_pts2

# #     def get_edge_pt(self, mask, pts):
# #         tmp_mask = mask[:, :, 0]
# #         x = pts[0]
# #         y = pts[1]
# #         edge_pts_x = []
# #         edge_pts_y = []
# #         for idx in range(len(x) - 1):
# #             if np.abs(tmp_mask[x[idx], y[idx]] - tmp_mask[x[idx + 1], y[idx + 1]]) == 1 or np.abs(
# #                     tmp_mask[x[idx], y[idx]] - tmp_mask[x[idx + 1], y[idx + 1]]) == 255:
# #                 edge_pts_x.append(x[idx])
# #                 edge_pts_y.append(y[idx])

# #         return edge_pts_x, edge_pts_y


# #     def measure_points_w_line(self,img, mask):
# #         list_xy = np.where(mask[:,:,0]== 255)
# #         tmp = np.shape(list_xy)
# #         # for idx in range(tmp[1]):
# #             # draw_pts(img, (list_xy[0][idx],list_xy[1][idx]),(255,0,0))
# #         return list_xy
# def Denormalization(image, mean=0.5, std=0.5):
#     return (image * std) + mean


# def Save(input, output, box, p_id, frame , roi, cfg , data_type, outputdir = None, foundation= False):
#     if not os.path.exists(outputdir):
#         mkdirs([outputdir])
        
#     pred = output["masks"]
#     pred = (pred>0.5).float()
    
#     gt = output["label"]
    
#     folder_path = os.path.join(outputdir, p_id)   
#     if not os.path.exists(folder_path):
#             mkdirs([folder_path])
            
#     input_image = input[0].detach().cpu().numpy()
#     input_image = np.expand_dims(input_image, axis=2)
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    
#     gt_mask = gt.detach().cpu().numpy().squeeze()
#     pred_mask = pred.detach().cpu().numpy().squeeze()
#     # c_box = bboxes.numpy().squeeze()
    
#     gt_mask = Image.fromarray( gt_mask.astype(np.uint8), mode="L")
#     gt_mask = F.resize(gt_mask, [256,256], interpolation=F.InterpolationMode.NEAREST)
#     gt_mask = np.array(gt_mask)

#     #check center of mask
#     input_image[int(box[0])-10 : int(box[0]) +10, int(box[1])-10 : int(box[1]) +10] = 255
#     input_image[int(box[2])-10 : int(box[2]) +10, int(box[3])-10 : int(box[3]) +10] = 255 
    
#     # pred_mask[int(box[0])-10 : int(box[0]) +10, int(box[1])-10 : int(box[1]) +10] = 3
#     # pred_mask[int(box[2])-10 : int(box[2]) +10, int(box[3])-10 : int(box[3]) +10] = 3
#     pred_mask[int(box[1])-10 : int(box[1]) +10, int(box[0])-10 : int(box[0]) +10] = 3
#     pred_mask[int(box[3])-10 : int(box[3]) +10, int(box[2])-10 : int(box[2]) +10] = 3 
    
#     # pred_mask[int(500)-10 : int(500) +10, int(200)-100 : int(200) +100] = 30
#     pred_mask = Image.fromarray( pred_mask.astype(np.uint8), mode="L")
#     pred_mask = F.resize(pred_mask, [256,256], interpolation=F.InterpolationMode.NEAREST)
#     pred_mask = np.array(pred_mask)
    
#     png_dir = os.path.join(folder_path, str(frame))
    
#     # input image
#     save_png(png_dir + "_" + roi + "_input.png", input_image)
#     # prediction mask
#     save_png(png_dir + "_" + roi + "_output.png", pred_mask)
#     # ground-truth mask
#     save_png(png_dir + "_" + roi + "_label.png", gt_mask)
#     # overlay image and mask
#     input_image_with_contour = draw_contour_2(
#         input_image.copy(), gt_mask, pred_mask, num_classes=1
#     )
#     save_png(png_dir + "_" + roi + "_input_image_contour.png", input_image_with_contour)
    
    
            
#     # prediction mask w prompts
#     save_png(png_dir + "_" + roi + "_output_w_prompts.png", pred_mask)
        
# def save_results(input_batch, output, cfg , data_type, fr, outputdir = None, foundation= False):
    
#     if cfg == False:
#         num_classes = len(output["prediction"][0])
        
#         file_name = input_batch["file_name"][0]
#         total_frame = np.shape(input_batch["input"])[2]
#         # inference_result_dir = set_dir_recursively(data_type, inference_result_dir)

#         dest_dir = "./resuilts_img"
#         dest_dir = os.path.join(dest_dir, cfg.trainer.model_type)
        
        
#         if not os.path.exists(dest_dir):
#             mkdirs([dest_dir])

#         for dcm_frame_name in range(total_frame):
#             folder_path = os.path.join(dest_dir, file_name)
#             pred_path = os.path.join(dest_dir, file_name, "result")
#             image_path = os.path.join(dest_dir, file_name, "image")
            
#             if not os.path.exists(folder_path):
#                 mkdirs([folder_path])
#                 mkdirs([pred_path])
#                 # mkdirs([image_path])
            
#             input_dir = os.path.join(image_path, str(dcm_frame_name))
#             result_dir = os.path.join(pred_path, str(dcm_frame_name))
        
#             # input_image = (Denormalization(input_batch["input"][0, 0, dcm_frame_name, :, :].cpu().numpy())).astype(dtype=np.uint8)

#             input_image = (input_batch["input"][0, 0, dcm_frame_name, :, :].cpu().numpy())*255
#             input_image = input_image.astype(dtype=np.uint8)
#             input_image = np.expand_dims(input_image, axis=2)
            
#             input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#             mask = output["prediction"][0].argmax(0)
#             model_prediction = mask[dcm_frame_name, :,:].cpu().numpy()

#             input_image_with_contour = draw_contour(
#                 input_image.copy(), input_batch["label"][0,dcm_frame_name,:,:].cpu().numpy(), model_prediction, num_classes=num_classes
#             )
#             save_png(input_dir + ".png", input_image)
#             save_png(result_dir + ".png", input_image_with_contour)
            
#     else:
        
#         dest_dir = outputdir
        
#         original_size = input_batch["original_size"]
#         patient_name = input_batch["patient_name"]
            
#         input_image = input_batch["image"][0,:,:].detach().cpu().numpy()
#         input_image = Image.fromarray( input_image.astype(np.uint8), mode="L") 
#         input_image = F.resize(input_image, original_size, interpolation=F.InterpolationMode.BILINEAR)
#         n_input_image = np.array(input_image)
#         input_image = np.expand_dims(n_input_image, axis=2)
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#         pred_mask = output
        
#         folder_path = os.path.join(dest_dir, patient_name)   
#         image_path = os.path.join(folder_path, "image")
#         pred_path = os.path.join(folder_path, "result")
    
#         if not os.path.exists(folder_path):
#             mkdirs([folder_path])
#             mkdirs([pred_path])
#             mkdirs([image_path])
        
#         if "label" in list(input_batch.keys()):
#             input_image_with_contour = draw_contour(
#                     input_image.copy(), input_batch["label"][0,dcm_frame_name,:,:].cpu().numpy(), pred_mask, num_classes=4)
#         else:
#             input_image_with_contour = draw_contour(
#                     input_image.copy(), None, pred_mask, num_classes=4)
        
#         save_png(os.path.join(image_path, str(fr) + "_.png"), input_image)
#         save_png(os.path.join(pred_path, str(fr) + "_.png"), input_image_with_contour)
            
#     return n_input_image, np.array(pred_mask, np.uint8)
#         # with GT
#         # input_image = input_batch["image"][0,:,:].detach().cpu().numpy()
#         # gt_mask = input_batch["mask"].argmax(0).detach().cpu().numpy().squeeze()
#         # pred_mask = output
        
#         # patient_name = input_batch["patient_name"]
#         # frame_num = str(input_batch["frame_num"])
#         # original_size = input_batch["original_size"],
#         # pts_prompt = input_batch["point_coords"]
        
        
#         # input_image = Image.fromarray( input_image.astype(np.uint8), mode="L") 
#         # input_image = F.resize(input_image, original_size[0], interpolation=F.InterpolationMode.BILINEAR)
        
#         # input_image = np.array(input_image)
#         # input_image = np.expand_dims(input_image, axis=2)
#         # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        
#         # folder_path = os.path.join(dest_dir, patient_name)   
        
#         # if not os.path.exists(folder_path):
#         #     mkdirs([folder_path])
            
#         # png_dir = os.path.join(folder_path, frame_num)
        
#         # # input image
#         # save_png(png_dir + "_input.png", input_image)
#         # # prediction mask
#         # save_png(png_dir + "_output.png", pred_mask)
#         # # ground-truth mask
#         # save_png(png_dir + "_label.png", gt_mask)
#         # # overlay image and mask
#         # input_image_with_contour = draw_contour(
#         #     input_image.copy(), gt_mask, pred_mask, num_classes=num_classes
#         # )
#         # save_png(png_dir + "_input_image_contour.png", input_image_with_contour)
        
#         # #check center of mask
#         # if data_type == "camus":
#         #     pred_mask[int(pts_prompt[0].numpy()[0][0])-1 : int(pts_prompt[0].numpy()[0][0]) +1, int(pts_prompt[0].numpy()[0][1])-1:int(pts_prompt[0].numpy()[0][1])+1] = 5
#         #     pred_mask[int(pts_prompt[1].numpy()[0][0])-1 : int(pts_prompt[1].numpy()[0][0]) +1, int(pts_prompt[1].numpy()[0][1])-1:int(pts_prompt[1].numpy()[0][1])+1] = 5
#         #     # pred_mask[int(pts_prompt[2].numpy()[0][0])-10 : int(pts_prompt[2].numpy()[0][0]) +10, int(pts_prompt[2].numpy()[0][1])-10:int(pts_prompt[2].numpy()[0][1])+10] = 5
#         #     # pred_mask[int(pts_prompt[3].numpy()[0][0])-10 : int(pts_prompt[3].numpy()[0][0]) +10, int(pts_prompt[3].numpy()[0][1])-10:int(pts_prompt[3].numpy()[0][1])+10] = 5 
#         # else:
#         #     pred_mask[int(pts_prompt[0].numpy()[0][0])-1 : int(pts_prompt[0].numpy()[0][0]) +1, int(pts_prompt[0].numpy()[0][1])-1:int(pts_prompt[0].numpy()[0][1])+1] = 5
#         #     pred_mask[int(pts_prompt[1].numpy()[0][0])-1 : int(pts_prompt[1].numpy()[0][0]) +1, int(pts_prompt[1].numpy()[0][1])-1:int(pts_prompt[1].numpy()[0][1])+1] = 5
        
#         # # prediction mask w prompts
#         # save_png(png_dir + "_output_w_prompts.png", pred_mask)
            
            
            
            
#         # for dcm_frame_name in range(total_frame):
#         #     folder_path = os.path.join(dest_dir, file_name)
            
#         #     if not os.path.exists(folder_path):
#         #         mkdirs([folder_path])
            
#         #     png_dir = os.path.join(folder_path, str(dcm_frame_name))
        
#         #     # input_image = (Denormalization(input_batch["input"][0, 0, dcm_frame_name, :, :].cpu().numpy())).astype(dtype=np.uint8)

#         #     input_image = (input_batch["input"][0, 0, dcm_frame_name, :, :].cpu().numpy())*255
#         #     input_image = input_image.astype(dtype=np.uint8)*255
#         #     input_image = np.expand_dims(input_image, axis=2)
            
#         #     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#         #     mask = output["prediction"][0].argmax(0)
#         #     model_prediction = mask[dcm_frame_name, :,:].cpu().numpy()

#         # # input image
#         #     save_png(png_dir + "_input.png", input_image)
#         # # prediction mask
#         #     save_png(png_dir + "_output.png", model_prediction)
#         # # ground-truth mask
#         #     save_png(png_dir + "_label.png", input_batch["label"][0,dcm_frame_name,:,:].cpu().numpy())
#         # # overlay image and mask
#         #     input_image_with_contour = draw_contour(
#         #         input_image.copy(), input_batch["label"][0,dcm_frame_name,:,:].cpu().numpy(), model_prediction, num_classes=num_classes
#         #     )
#         #     save_png(png_dir + "_input_image_contour.png", input_image_with_contour)





#     # npy_dir = os.path.join(inference_result_dir, data_type, file_name, "npy")
    
#     # mkdirs([png_dir, npy_dir])
    
#     # npy_dir = os.path.join(npy_dir, dcm_frame_name)

#     # uncertainty_handler = UncertaintyHandler(output["prediction"].exp().cpu())
#     # try:
#     #     setattr(uncertainty_handler, "MC_predictions", output["MC_predictions"].exp().cpu())
#     # except:
#     # #     pass
    
#     # if cfg.data.datatype != "video":
#     #     input_image = (Denormalization(input_batch["input"][0, 0, :, :].cpu().numpy()) * 255).astype(dtype=np.uint8)
#     # else:
#     #     input_image = (Denormalization(input_batch["input"][0, 0, :, :].cpu().numpy())).astype(dtype=np.uint8)
#     #     input_image = input_image[0, :, :]
        
#     # input_image = np.expand_dims(input_image, axis=2)
#     # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#     # model_prediction = output["prediction"][0].argmax(0).cpu().numpy()

#     # # input image
#     # # if cfg.logging.inference_analysis.idnput:
#     # save_png(png_dir + "_input.png", input_image)

#     # # ground-truth mask
#     # # if cfg.logging.inference_analysis.label and cfg.data.datatype != "video":
#     # save_png(png_dir + "_label.png", input_batch["label"][0].cpu().numpy())
#     #     # save_npy(npy_dir + "_label.npy", input_batch["label"][0].cpu().numpy())
#     # # elif cfg.logging.inference_analysis.label and cfg.data.datatype == "video":
#     # # save_png(png_dir + "_label.png", input_batch["label"][0][0].cpu().numpy())
#     #     # save_npy(npy_dir + "_label.npy", input_batch["label"][0][0].cpu().numpy())

#     # # model output
#     # # if cfg.logging.inference_analysis.output:
#     # model_prediction = output["prediction"][0].argmax(0).cpu().numpy()
#     #     # model_prediction = output["prediction"][0].cpu().numpy()
#     #     # for i in range(model_prediction):
#     #     #     model_prediction
#     # save_png(png_dir + "_output.png", model_prediction)
#     #     # save_npy(npy_dir + "_output.npy", model_prediction)

#     # # draw contour1 of the model output
#     # # if cfg.logging.inference_analysis.input_image_contour and cfg.data.datatype != "video":
#     # input_image_with_contour = draw_contour(
#     #     input_image.copy(), input_batch["label"][0].cpu().numpy(), model_prediction, num_classes=num_classes
#     # )
#     # save_png(png_dir + "_input_image_contour.png", input_image_with_contour)
#     # # elif cfg.logging.inference_analysis.input_image_contour and cfg.data.datatype == "video":
#     # input_image_with_contour = draw_contour(
#     #     input_image.copy(), input_batch["label"][0][0].cpu().numpy(), model_prediction, num_classes=num_classes
#     # )
#     # save_png(png_dir + "_input_image_contour.png", input_image_with_contour)

#     # probability map of model output
#     # if cfg.logging.inference_analysis.probability_map:
#     #     save_png(png_dir + "_probability_map.png", output["prediction"][0, 1].cpu().numpy())

#     # # CCA
#     # if cfg.logging.inference_analysis.CCA:
#     #     model_prediction_CCA = CCA(model_prediction, num_classes)
#     #     save_png(png_dir + "_output_w_CCA.png", model_prediction_CCA)

#     ##############################################################
#     ## Uncertainty Metrics. Occasionally it requires MC samples ##
#     ##############################################################

#     # # entropy
#     # if cfg.logging.inference_analysis.entropy:
#     #     entropy = uncertainty_handler.get_entropy()
#     #     save_png(png_dir + "_entropy.png", entropy)

#     # # BALD
#     # if cfg.logging.inference_analysis.BALD:
#     #     BALD = uncertainty_handler.get_BALD()
#     #     save_png(png_dir + "_BALD.png", BALD)

#     # # separate epistemic and aleatoric
#     # if cfg.logging.inference_analysis.epistemic:
#     #     aleatoric, epistemic = uncertainty_handler.separate_uncertainty()
#     #     tr_epistemic, tr_aleatoric = epistemic.sum(0), aleatoric.sum(0)
#     #     save_png(png_dir + "_epistemic.png", tr_epistemic), save_png(png_dir + "_aleatoric.png", tr_aleatoric)


# def save_results_all_frame(cfg, inference_result_dir, data_type, patient_name, frames, input, output, original_shape, area_dict):
#     num_classes = len(output[0])
#     input_dir = os.path.join(inference_result_dir, data_type, patient_name, "input")
#     output_dir = os.path.join(inference_result_dir, data_type, patient_name, "output")
#     contour_dir = os.path.join(inference_result_dir, data_type, patient_name, "contour1")
#     prob_dir = os.path.join(inference_result_dir, data_type, patient_name, "prob")
#     mkdirs([input_dir, output_dir, contour_dir])
#     print(input_dir)
#     list_lv_area = []
#     for idx, frame in enumerate(frames):
#         frame = frame.numpy()
#         # input_image = (Denormalization(input[idx, 0, :, :].cpu().numpy()) * 255).astype(dtype=np.uint8)

#         input_image = cv2.cvtColor(input[idx, :, :].cpu().numpy(), cv2.COLOR_GRAY2RGB)
#         model_prediction = output[idx][:].argmax(0).cpu().numpy()
#         print (np.shape(output[0]))
#         npy_dir = os.path.join(inference_result_dir, data_type, patient_name, "npy")
#         mkdirs([input_dir, output_dir, contour_dir, npy_dir])
#         # input images
#         # if cfg.logging.inference_analysis.input:
#         #     save_png(os.path.join(input_dir, "{}.png".format(str(frame))), input_image)

#         # model output
#         if cfg.logging.inference_analysis.output:
#             save_png(os.path.join(output_dir, "{}.png".format(str(frame))), model_prediction)

#         # if cfg.logging.inference_analysis.output:
#         #
#         #     save_png(os.path.join(output_dir, "{}.png".format(str(frame))), model_prediction)

#         if cfg.logging.inference_analysis.probability_map:
#             save_npy(os.path.join(npy_dir, "{}.npy".format(frame)), output.cpu().numpy())
#             # for ch_idx in range(num_classes):
#             #     print ('---'*100, np.shape(output[idx]))
#             #     print (np.shape(output[idx][ch_idx]))

#                 # save_png(os.path.join(output_dir,"_probability_map_"+ str(ch_idx)+ "{}.png".format(str(frame))), output[idx][ch_idx].cpu().numpy())

#         # draw contour1 of the model output
#         if cfg.logging.inference_analysis.input_image_contour:
#             print (num_classes)
#             input_image_with_contour, list_area = draw_contour_prediction(
#                 input_image.copy(), model_prediction, num_classes=num_classes
#             )
#             list_lv_area.append(list_area[1])
#             # area_dict["LA"].append(list_area[0])
#             # area_dict["LV"].append(list_area[1])
#             # area_dict["RA"].append(list_area[2])
#             # area_dict["RV"].append(list_area[3])
#             # area_dict["LVwall"].append(list_area[4])
#             save_png(os.path.join(contour_dir, "{}.png".format(frame)), input_image_with_contour)

# def save_results_all_frame_hybrid_plax(cfg, inference_result_dir, data_type, patient_name, frames, input, output, output_detect,
#                            original_shape, area_dict):
#     num_classes = len(output[0])
#     num_classes_detect = len(output_detect[0])

#     input_dir = os.path.join(inference_result_dir, data_type, patient_name, "input")
#     output_dir = os.path.join(inference_result_dir, data_type, patient_name, "output")
#     contour_dir = os.path.join(inference_result_dir, data_type, patient_name, "contour1")
#     prob_dir = os.path.join(inference_result_dir, data_type, patient_name, "prob")

#     mkdirs([input_dir, output_dir, contour_dir])
#     print(input_dir)
#     list_lv_area = []
#     for idx, frame in enumerate(frames):
#         frame = frame.numpy()
#         # input_image = (Denormalization(input[idx, 0, :, :].cpu().numpy()) * 255).astype(dtype=np.uint8)

#         input_image = cv2.cvtColor(input[idx, :, :].cpu().numpy(), cv2.COLOR_GRAY2RGB)
#         model_prediction = output[idx][:].argmax(0).cpu().numpy()
#         model_prediction_detect = output_detect[idx][:].argmax(0).cpu().numpy()


#         print(np.shape(output[0]))
#         npy_dir = os.path.join(inference_result_dir, data_type, patient_name, "npy")
#         mkdirs([input_dir, output_dir, contour_dir, npy_dir])
#         # input images
#         # if cfg.logging.inference_analysis.input:
#         #     save_png(os.path.join(input_dir, "{}.png".format(str(frame))), input_image)

#         # model output
#         if cfg.logging.inference_analysis.output:
#             save_png(os.path.join(output_dir, "{}.png".format(str(frame))), model_prediction)

#         # if cfg.logging.inference_analysis.output:
#         #
#         #     save_png(os.path.join(output_dir, "{}.png".format(str(frame))), model_prediction)

#         if cfg.logging.inference_analysis.probability_map:
#             save_npy(os.path.join(npy_dir, "{}.npy".format(frame)), output.cpu().numpy())
#             # for ch_idx in range(num_classes):
#             #     print ('---'*100, np.shape(output[idx]))
#             #     print (np.shape(output[idx][ch_idx]))

#             # save_png(os.path.join(output_dir,"_probability_map_"+ str(ch_idx)+ "{}.png".format(str(frame))), output[idx][ch_idx].cpu().numpy())

#         # draw contour1 of the model output
#         if cfg.logging.inference_analysis.input_image_contour:
#             input_image_with_contour, list_area = draw_contour_prediction(
#                 input_image.copy(), model_prediction, num_classes=num_classes
#             )
#             # input_image_with_contour, list_area = draw_contour_prediction(
#             #     input_image.copy(), model_prediction_detect, num_classes=num_classes_detect
#             # )

#             input_image_with_contour_detect, list_area_detect = draw_contour_prediction_line(
#                 input_image.copy(), model_prediction_detect, num_classes=num_classes_detect, line = True
#             )

#             # list_lv_area.append(list_area[1])
#             # area_dict["LA"].append(list_area[0])
#             # area_dict["LV"].append(list_area[1])
#             # area_dict["RA"].append(list_area[2])
#             # area_dict["RV"].append(list_area[3])
#             # area_dict["LVwall"].append(list_area[4])
#             # save_png(os.path.join(contour_dir, "{}.png".format(frame)), input_image_with_contour)
#             save_png(os.path.join(contour_dir, "{}_.png".format(frame)), input_image_with_contour_detect)


#     # video
#     # print("inference all frames_ video")
#     # num_classes = len(output[0])
#     #
#     # input_dir = os.path.join(inference_result_dir, data_type, patient_name, "input")
#     # output_dir = os.path.join(inference_result_dir, data_type, patient_name, "output")
#     # contour_dir = os.path.join(inference_result_dir, data_type, patient_name, "contour1")
#     # npy_dir = os.path.join(inference_result_dir, data_type, patient_name, "npy")
#     # mkdirs([input_dir, output_dir, contour_dir, npy_dir])
#     #
#     # for idx, frame in enumerate(frames):
#     #     frame = frame.numpy()
#     #     input_image = (Denormalization(input[idx, 0, 0, :, :].cpu().numpy())).astype(dtype=np.uint8)
#     #     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#     #     model_prediction = output[idx].argmax(0).cpu().numpy()
#     #     # input image
#     #     if cfg.logging.inference_analysis.input:
#     #         save_png(os.path.join(input_dir, "{}.png".format(str(frame))), input_image)
#     #
#     #     # model output
#     #     if cfg.logging.inference_analysis.output:
#     #         save_png(os.path.join(output_dir, "{}.png".format(str(frame))), model_prediction)
#     #
#     #     # draw contour1 of the model output
#     #     if cfg.logging.inference_analysis.input_image_contour:
#     #         input_image_with_contour = draw_contour_prediction(
#     #             input_image.copy(), model_prediction, num_classes=num_classes
#     #         )
#     #         save_png(os.path.join(contour_dir, "{}.png".format(frame)), input_image_with_contour)
#     #
#     #     save_npy(os.path.join(npy_dir, "{}.npy".format(frame)), model_prediction)
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
