import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology


def plot_images_for_comparison(image):
    original = cv2.imread('test_images/' + image, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay = cv2.imread('segmentation/Output_Prediction/test_4/Overlay/' + image)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('test_masks/' + image,0)
    prediction = cv2.imread('segmentation/Output_Prediction/test_4/Label/' + image,0)

    plt.figure(figsize=(15,15))
    plt.imshow(overlay)
    plt.subplot(2,2,1),plt.imshow(original)
    plt.subplot(2,2,2),plt.imshow(overlay)
    plt.subplot(2,2,3),plt.imshow(mask)
    plt.subplot(2,2,4),plt.imshow(prediction)
#     plt.imshow(original)
    return (original, mask, overlay, prediction)


def calculate_iou(prediction, ground_truth):
    ClassIOU=np.zeros(2)#Vector that Contain IOU per class
    ClassWeight=np.zeros(2)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(2): # Go over all classes
        Intersection=np.float32(np.sum((prediction==ground_truth)*(prediction==i)))# Calculate class intersection
        Union=np.sum(prediction==i)+np.sum(prediction==i)-Intersection # Calculate class Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union
        else:
            print(f"No pixels in class {i}")
    for i in range(2):
        print(f"class{i}:", str(ClassIOU[i]))
        print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
        print("Image Predicition Accuracy)" + str(np.float32(np.sum(ground_truth == prediction)) / ground_truth.size))
        
    return ClassIOU, ClassWeight


def post_process(mask, original):
    kernel = np.ones((10,10 ),np.uint8)
    dilation = cv2.dilate(mask, kernel)
    new_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(new_image,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,8)
    mask_cut = cv2.bitwise_and(cv2.bitwise_not(binarized), dilation*255)

    # mask = cv2.cvtColor(mask_cut, cv2.COLOR_GRAY2BGR)
    processed = morphology.remove_small_objects(mask_cut.astype(bool), min_size=20, connectivity=10)
    return processed

