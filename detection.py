import numpy as np
import os
import sys
import tensorflow as tf
import time
import cv2
import statistics
 
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
 
# Path to models
# MODEL_PATH = 'frcnn_models'
MODEL_PATH = 'frozen_models/mrcnn_models'
 
 
########### ++++++++++++++++++++++ ###########
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# mod = '_resnet101'
mod = '_resnet50'
# mod = '_inception'
# mod = '_inception_resnet'
 
################ FRCNN ################ 
 
 
################  MRCNN ################ 
MODEL_NAME = 'balloon' + mod + '.pb'
 
########### ++++++++++++++++++++++ ###########
PATH_TO_CKPT = os.path.join(MODEL_PATH, MODEL_NAME) 
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'balloon_label_map.pbtxt')
 
NUM_CLASSES = 1
 
# LOAD FROZEN TENSORFLOW MODEL INTO MEMORY
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
 
# LOADING LABEL MAP
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
#HELPER CODE
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
        # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        
            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
 
def run_inference_for_multiple_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []
            dict_time = []
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, images[0].shape[0], images[0].shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            for image in images:
                # Run inference
                start = time.time()
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
                end = time.time()
                print('inference time : {}'.format(end - start))
 
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
 
                output_dict_array.append(output_dict)
                dict_time.append(end - start)
    return output_dict_array, dict_time
 
# DETECTION
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
 
# subdir_name = os.listdir(PATH_TO_TEST_IMAGES_DIR)
subdir_name = ['balloon-test']
 
PATH_TO_RESULT_IMAGES_DIR = 'result_images'
 
min_score_thresh = 0.5
for path in subdir_name:
    print('Path: {}'.format(path))
    # Listing the test images
    subdir_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, path)
    (_, _, filenames) = next(os.walk(subdir_path))
 
    # Listing the result images
    result_img_path = os.path.join(PATH_TO_RESULT_IMAGES_DIR, path + mod)
    # result_mask_path = os.path.join(PATH_TO_RESULT_IMAGES_DIR, path + mod + '_mask')

    # Creating folder for result images
    if not os.path.exists(result_img_path):
        os.makedirs(result_img_path)
        # os.makedirs(result_mask_path)
    (_, _, filenames_result) = next(os.walk(result_img_path))
 
    # Set file difference if any
    diff_files = np.setdiff1d(filenames, filenames_result)
    print('Total different files: {}'.format(len(diff_files)))
 
    # Process the images by setting up batches
    batch_size = 10
    chunks = len(diff_files) // batch_size + 1
    ave_time = []
    for i in range(chunks):
        batch = diff_files[i * batch_size: (i + 1) * batch_size]
        images = []
        files = []
        proc_time = []
        for file in batch:
            image_path = os.path.join(subdir_path, file)
            print('Reading file {}'.format(image_path))
            image = cv2.imread(image_path)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image_np)
            files.append(file)
 
        output_dicts, out_time = run_inference_for_multiple_images(images, detection_graph)
        print('length of output_dicts is : {}'.format(len(output_dicts)))
        if len(output_dicts) == 0:
            break
 
        for idx in range(len(output_dicts)):
            output_dict = output_dicts[idx]
            image_np = images[idx]
            file = files[idx]

            #  SAVING BOUNDING BOX AND CONFIDENCE RATE
            boxes = output_dict['detection_boxes']
            classes = output_dict['detection_classes']
            scores = output_dict['detection_scores']
            masks = output_dict['detection_masks']

            # Visualization of the results of a detection.
            start = time.time()
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              boxes,
              classes,
              scores,
              category_index,
              instance_masks=masks, # output_dict.get('detection_masks'),
              use_normalized_coordinates=True, 
              min_score_thresh=min_score_thresh,
              line_thickness=4)
 
            height, width, chan = image_np.shape
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_img_path, file), image_np)
            print('Saving {}, time : {}'.format(file, time.time()-start))
 
            proc_time.append(time.time()-start + out_time[idx])
            # count += 1
 
            # VISUALIZING MASK IMAGES (COMMENT THIS SECTION FOR FRCNN)
            # if output_dict.get('detection_masks') is not None:
            #   inst_masks = output_dict.get('detection_masks')
            #   mask_img = np.zeros((height, width, 3), np.uint8)
            #   label_img = np.zeros((height, width ), np.uint8)
            #   for j in range(0, inst_masks.shape[0]):
            #     if scores[j] >= min_score_thresh:
            #         label_img += inst_masks[j]
            #   mask_img[label_img > 0] = [255, 255, 255]
            #   cv2.imwrite(os.path.join(result_mask_path, file), mask_img)
             
            # WRITING STATISTICS RESULT
            # for index, value in enumerate(classes):
            #     if scores[index] >= min_score_thresh:
            #         ymin, xmin, ymax, xmax = boxes[index]
            #         f.write(file + ',' + str(width) + ',' + str(height) + ',crack,' + \
            #             str(int(xmin * width)) + ',' + str(int(ymin * height)) + ',' + \
            #             str(int(xmax * width)) + ',' + str(int(ymax * height)) + ',' + str(scores[index]) + '\n')

        if len(proc_time) != 0:
            mean_batch_time = statistics.mean(proc_time)
            print('mean one-batch processing time: {}'.format(mean_batch_time))
            ave_time.append(mean_batch_time)
        images.clear()
        files.clear()
        proc_time.clear()
        output_dicts.clear()
        out_time.clear()
 
    # f.close()
    if len(ave_time) != 0:
        print('mean total processing time: {}'.format(statistics.mean(ave_time)))
