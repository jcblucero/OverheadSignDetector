#
#   Load the trained neural network and perform object on either video or image
#   Meant to be run from v1.13 of tensorflow/models/research/object_detection
#
#
## Some of this code is taken from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## and modified for my use case

import numpy as np
import os
import sys
import tensorflow as tf
import cv2 as cv

#Object detection utilities import
from utils import label_map_util
from utils import visualization_utils as vis_util

#Set up saved model location
SAVED_MODEL_FOLDER = 'inference_graph'
NUM_CLASSES = 11
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_INF_GRAPH = SAVED_MODEL_FOLDER + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')

#Load the frozen tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_INF_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#Load the label map
#Labelmaps map index to a class name (string)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Specific Overhead sign class to search for
OVERHEAD_SIGN_CLASS = 11

#Indices into Boxes returned by Tensorflow
Y_MIN_INDEX = 0
X_MIN_INDEX = 1
Y_MAX_INDEX = 2
X_MAX_INDEX = 3

#Load a session based on graph made from trained model
sess = tf.Session(graph=detection_graph)
#Define the input tensor
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Takes as input a numpy image (ndarry of [x,y,3] in bgr)
# Detects objects using trained detector and return as tuple of (boxes, scores, classes, num)
# returned objects are ndarrays
def perform_detection(np_image):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    rgb_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(rgb_image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    #np.squeeze reduces single dimensions/unused dimensions
    return (np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), num)

#determine the height of the sign from the detected digits in the image
#return the boxes and classes or the sign height as lists
def determine_sign_height(sign_box,digit_boxes,digit_classes):

    sign_box_x_min = sign_box[X_MIN_INDEX]
    sign_box_x_max = sign_box[X_MAX_INDEX]
    sign_box_width = sign_box_x_max - sign_box_x_min

    #Interim lists we will build up
    merged_boxes_list = []
    merged_classes_list = []

    #First find the digits which reside inside the sign's box
    #We only look at x bounds for this under assumption boxes will never be vertically stacked
    #Check that x_min of digits is > x_min of sign, and x_max of digits < x_max of sign
    selection_indices = np.where( (digit_boxes[:,X_MIN_INDEX] > sign_box_x_min)
                                   & (digit_boxes[:,X_MAX_INDEX] < sign_box_x_max) )
    selected_digit_boxes = digit_boxes[selection_indices]
    selected_classes = digit_classes[selection_indices]

    #Merge digits that are close to each other to form numbers up to two digits in length
    #closeness is determined as within some threshold of sign_box_width
    threshold = sign_box_width * 0.05 #within 5% of box width of each other
    num_digits = selected_classes.size
    merged = np.zeros(num_digits)

    for index in range(num_digits):

        #Skip if it's already been merged
        if(merged[index]):
            continue

        digit_x_max = selected_digit_boxes[index,X_MAX_INDEX] + threshold
        digit_box = selected_digit_boxes[index]
        #Find indices where candidate's left side is less than digit's right side, and candidates right side exceeds digit's right side
        merging_indices = np.where((selected_digit_boxes[:,X_MIN_INDEX] < digit_x_max) & (selected_digit_boxes[:,X_MAX_INDEX] > digit_x_max))
        #merging_indices is a list of (ndarray,dtype)
        if(merging_indices[0].size >= 1):
            #Only keep the first digit that meets our criteria
            merging_index = merging_indices[0][0]
            #mark as merged
            merged[merging_index] = 1
            merged[index] = 1

            #Special case for 0's, because 0 is reserved in tensorflow labelmaps, so we had to use 10.
            if( selected_classes[merging_index] == 10):
                merging_digit = 0
            else:
                merging_digit = selected_classes[merging_index]
            new_number = selected_classes[index] * 10 + merging_digit
            merging_box = selected_digit_boxes[merging_index]

            #Create a new box for new number
            min_array = np.minimum(digit_box,merging_box)
            max_array = np.maximum(digit_box,merging_box)
            merged_boxes_list.append( [min_array[Y_MIN_INDEX],min_array[X_MIN_INDEX],max_array[Y_MAX_INDEX],max_array[X_MAX_INDEX]] )

            merged_classes_list.append( new_number )
        else:
            pass
            #Do nothing, we will append unmerged digits after examining all digits

    for index in range(num_digits):
        digit_box = selected_digit_boxes[index]
        if(merged[index] == 0):
            merged_boxes_list.append( [digit_box[Y_MIN_INDEX],digit_box[X_MIN_INDEX],digit_box[Y_MAX_INDEX],digit_box[X_MAX_INDEX]])

            #For our merged classes, 10 and 0 will overlap.
            #This is because 0 is reserved in Tensorflow labelmaps so we used 10 (since there is no 10 when only considering 1 digit)
            #But now we will actually have 2 digits, so we will map 0's back to 0 in our merged classes, and 10 to 10
            if( selected_classes[index] == 10):
                merged_classes_list.append(0)
            else:
                merged_classes_list.append(selected_classes[index])

    return (merged_boxes_list,merged_classes_list)

#Build a list of strings that will be displayed with the overhead signs
#input:
#   merged_classes_list - list of integers associated with the overhead sign
#output:
#   disp_str_list - list of strings to display with overhead sign
def build_overhead_disp_str_list(merged_classes_list):
    disp_str_list = ["Overhead Sign"]

    if( len(merged_classes_list) > 0 ):
        disp_str_list.append( "Feet: {}".format(merged_classes_list[0]) )
    else:
        disp_str_list.append( "Feet: N/A" )
    if( len(merged_classes_list) > 1):
        disp_str_list.append( "Inches: {}".format(merged_classes_list[1]) )
    else:
        disp_str_list.append( "Inches: N/A" )

    return disp_str_list

#determine the height displayed on overhead sign based on bounding boxes/number classes detected
# Inputs - (boxes,scores,classes,num) as ndarrays returned from object detection (perform_detection)
# Outputs:
#   overhead_sign_boxes - 2d ndarray where each row is (ymin,xmin,ymax,xmax) of a box
#   list_of_disp_str_list - list of strings to display for box, defining height of overhead sign
def overhead_classification(boxes,scores,classes,num):

    #Get the overhead boxes that meet our threshold
    selection_indices = np.where( (classes == OVERHEAD_SIGN_CLASS) & (scores >= 0.6) )
    overhead_boxes = boxes[selection_indices]
    overhead_classes = classes[selection_indices]

    #Get all digits that meet our threshold (0-9)
    selection_indices = np.where( (classes != OVERHEAD_SIGN_CLASS) & (scores >= 0.6) )
    digit_boxes = boxes[selection_indices]
    digit_classes = classes[selection_indices]

    #Here we determine all sign's heights
    #print("overhead boxes shape",overhead_boxes.shape,overhead_boxes.size)
    list_of_disp_str_list = []
    for index in range(overhead_boxes.shape[0]):
        (merged_digit_boxes_list,merged_classes_list) = determine_sign_height(overhead_boxes[index],digit_boxes,digit_classes)
        disp_str_list = build_overhead_disp_str_list(merged_classes_list)
        list_of_disp_str_list.append(disp_str_list)

    return overhead_boxes,list_of_disp_str_list

#Draw overhead boxes and labels on image
#Inputs:
#   image - image to draw on as ndarray
#   overhead_boxes - ndarray of boxes, where each box is defined by ndarray of [ymin,xmin,ymax,xmax]
#   list_of_disp_str_lists - list of, list of strings. One list of strings for each box
#Outputs:
#   image - ndarray image with boxes/labels drawn on
def draw_overhead_classification(image,overhead_boxes,list_of_disp_str_list):

    for index in range(overhead_boxes.shape[0]):

        box = overhead_boxes[index]
        vis_util.draw_bounding_box_on_image_array(
            image,
            box[Y_MIN_INDEX],#ymin,
            box[X_MIN_INDEX],#xmin,
            box[Y_MAX_INDEX],#ymax,
            box[X_MAX_INDEX],#xmax,
            #color='red',
            #thickness=4,
            #display_str_list=(),
            display_str_list = list_of_disp_str_list[index],
            use_normalized_coordinates=True)

    return image

#Given path to image
#Open image, run detector on it, and draw bounding box on it
#This will visualize ALL boxes/classes/scores over threshold
#return image as ndarray
def visualize_all_classes(path_to_image):
    image = cv.imread(path_to_image)

    (boxes, scores, classes, num) = perform_detection(image)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.60)

    return image

#Given path to image
#Open image, run detector on it, and draw bounding box on
#overhead sign class only. Apply labels to image
def visualize_overhead_single_image(path_to_image):
    image = cv.imread(path_to_image)

    #Detection and classification on image
    (boxes, scores, classes, num) = perform_detection(image)
    overhead_boxes,list_of_disp_str_list = overhead_classification(boxes,scores,classes,num)
    draw_overhead_classification(image,overhead_boxes,list_of_disp_str_list)

    return image

#Create an opencv video writer for .avi files in mp4 format
def build_mp4_video_writer(filename,frame_size,fps=20):
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    return cv.VideoWriter(filename,fourcc,fps,frame_size)

def set_image_240p(image):
    image = cv.resize(src=image,dsize=(426,240),interpolation=cv.INTER_AREA)
    return image

def rotate_90(image):
    new_image = cv.transpose(image)
    new_image = cv.flip(new_image,1)
    return new_image


#Set image to how it was trained on (smaller size, and correct sign orientation)
def set_image_to_trained_size(image):
    image = set_image_240p(image)
    return rotate_90(image)

def detect_and_write_video(file):
    cap = cv.VideoCapture(file)
    ret, frame = cap.read()
    frame = set_image_to_trained_size(frame)

    h,w,d = frame.shape
    video_out = build_mp4_video_writer('Test_Detection_Video.avi', (w,h), 20)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = set_image_to_trained_size(frame)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #Detection and classification on image
        (boxes, scores, classes, num) = perform_detection(frame)
        overhead_boxes,list_of_disp_str_list = overhead_classification(boxes,scores,classes,num)
        draw_overhead_classification(frame,overhead_boxes,list_of_disp_str_list)

        cv.imshow('frame', frame)
        video_out.write(frame)
        if cv.waitKey(10) == ord('q'):
            break

    cap.release()
    video_out.release()
    cv.destroyAllWindows()

#Given a path to a directory of images
#run object detection on images and write to a video
def detect_and_write_images_to_video(path_to_images):

    #Get all the files in the director
    for root,dirs,files in os.walk(path_to_images):
        pass
        break

    #Create the video writer
    image = cv.imread(os.path.join(root,files[0]))

    h, w, d = image.shape
    fps = 20
    video_out = build_mp4_video_writer('TEST_DETECTION.avi',(w,h),fps)

    #Loop through all files, run detection, and write to video out
    seconds_to_show = 2
    for file in files:
        image_path = os.path.join(root,file)
        image = cv.imread(image_path)
        image = cv.resize(image,(426,240))

        #Detection and classification on image
        (boxes, scores, classes, num) = perform_detection(image)
        overhead_boxes,list_of_disp_str_list = overhead_classification(boxes,scores,classes,num)
        draw_overhead_classification(image,overhead_boxes,list_of_disp_str_list)

        #show for x seconds in video
        for i in range(fps * seconds_to_show):
            video_out.write(image)

    #release VideoWriter
    video_out.release()

if __name__ == "__main__":
    #image_path = "test_image.PNG"
    image_path = "images/test/OverpassHeight_Three_Signs_2.PNG"
    image = visualize_overhead_single_image(image_path)
    cv.imwrite("OverpassHeight_Three_Signs_Test_Out.PNG",image)
    cv.imshow("Detection",image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #detect_and_write_video('images/Videos/VID_20200404_164046507_HDR.mp4')





