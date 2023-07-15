"""
Prediction part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""
from config import *
import matplotlib.pyplot as plt
from train import KaggleDataset
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion
from utils import non_max_suppression, extract_bboxes
import cv2
import random
import pandas as pd
from metrics import mean_iou
from tqdm import tqdm
import numpy as np
import joblib
plt.switch_backend('agg')


col_dict = {
    0:'g',
    1 : 'b',
    2 : '#FF8C00',
    3 : 'r'
}
def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

def plot_boundary(image, true_masks=None, pred_masks=None,classes=None, ax=None):
    """
    Plots provided boundaries of nuclei for a given image.
    """
    if ax is None:
        n_rows = 1
        n_cols = 1

        fig = plt.figure(figsize=[4*n_cols, int(4*n_rows)])
        gs = gridspec.GridSpec(n_rows, n_cols)

        ax = fig.add_subplot(gs[0])

    ax.imshow(image)
    for i in range(pred_masks.shape[-1]):
        contours = find_contours(pred_masks[..., i], 0.5, fully_connected='high')

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=col_dict[classes[i]])

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)  # aspect ratio of 1



def get_model(config, model_path=None):

    """
    Loads and returns MaskRCNN model for a given config and weights.
    """
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_path is None:
        model_path = model.find_last()[1]
        try:
            os.rename(model_path, model_path)
            print('Access on file ' + model_path + ' is available!')
            from shutil import copyfile
            dst = '../data/mask_rcnn_temp.h5'
            copyfile(model_path, dst)
            model_path = dst
        except OSError as e:
            print('Access-error on file "' + model_path + '"! \n' + str(e))

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)

    model.load_weights(model_path, by_name=True)

    return model


def ensemble_prediction(model, config, image):

    """ Test time augmentation method using non-maximum supression"""

    masks = []
    scores = []
    boxes = []

    results = {}

    result = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    masks.append(result['masks'])
    scores.append(result['scores'])
    boxes.append(extract_bboxes(result['masks']))

    temp_img = np.fliplr(image)
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.fliplr(result['masks'])
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    temp_img = np.flipud(image)
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.flipud(result['masks'])
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    angle = np.random.choice([1, -1])
    temp_img = np.rot90(image, k=angle, axes=(0, 1))
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.rot90(result['masks'], k=-angle, axes=(0, 1))
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    masks = np.concatenate(masks, axis=-1)
    scores = np.concatenate(scores, axis=-1)
    boxes = np.concatenate(boxes, axis=0)

    # config.DETECTION_NMS_THRESHOLD)
    keep_ind = non_max_suppression(boxes, scores, 0.1)
    masks = masks[:, :, keep_ind]
    scores = scores[keep_ind]

    results['masks'] = masks
    results['scores'] = scores

    return results


def cluster_prediction(model, config, image):

    """ Test time augmentation method using bounding box IoU"""
    # from utils import non_max_suppression, extract_bboxes, compute_overlaps
    height, width = image.shape[:2]

    # Predict masks on actual image
    result1 = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    # Handles no mask predictions
    if result1['masks'].shape[0] == 0:
        result1['masks'] = np.zeros([height, width, 1])
        result1['masks'][0, 0, 0] = 1
        result1['scores'] = np.ones(1)

    # Predict masks on LR flipped image
    temp_img = np.fliplr(image)
    result2 = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    result2['masks'] = np.fliplr(result2['masks'])
    # Handles no mask predictions
    if result2['masks'].shape[0] == 0:
        result2['masks'] = np.zeros([height, width, 1])
        result2['masks'][0, 0, 0] = 1
        result2['scores'] = np.ones(1)

    # Compute IoU on masks
    overlaps = utils.compute_overlaps_masks(result1['masks'], result2['masks'])
    for mm in range(overlaps.shape[0]):

        if np.max(overlaps[mm]) > 0.1:
            ind = np.argmax(overlaps[mm])
            mask = result1['masks'][:, :, mm] + result2['masks'][:, :, ind]
            result1['masks'][:, :, mm] = (mask > 0).astype(np.uint8)
            # result1['scores'][mm] = 0.5*(result1['scores'][mm]+result2['scores'][ind])
        else:
            result1['masks'][:, :, mm] = 0
    return result1


def postprocess_masks(result, image, min_nuc_size=10):

    """Clean overlaps between bounding boxes, fill small holes, smooth boundaries"""

    height, width = image.shape[:2]

    # If there is no mask prediction do the following
    if result['masks'].shape[0] == 0:
        result['masks'] = np.zeros([height, width, 1])
        result['masks'][0, 0, 0] = 1
        result['scores'] = np.ones(1)

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    if len(keep_ind) < result['masks'].shape[-1]:
        # print('Deleting',len(result['masks'])-len(keep_ind), ' empty result['masks']')
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]

    sort_ind = np.argsort(result['scores'])[::-1]
    result['masks'] = result['masks'][..., sort_ind]
    overlap = np.zeros([height, width])

    # Removes overlaps from masks with lower score
    for mm in range(result['masks'].shape[-1]):
        # Fill holes inside the mask
        mask = binary_fill_holes(result['masks'][..., mm]).astype(np.uint8)
        # Smoothen edges using dilation and erosion
        mask = erosion(dilation(mask))
        # Delete overlaps
        overlap += mask
        mask[overlap > 1] = 0
        out_label = label(mask)
        # Remove all the pieces if there are more than one pieces
        if out_label.max() > 1:
            mask[()] = 0

        result['masks'][..., mm] = mask

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    if len(keep_ind) < result['masks'].shape[-1]:
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]

    return result



def pred_n_plot_test(model, config, test_path='../data/stage2_test_final/', save_plots=False):
    """
    Predicts nuclei for each image, draws the boundaries and saves in images folder.

    """

    # Load test dataset
    test_ids = os.listdir(test_path)
    dataset_test = KaggleDataset()
    dataset_test.load_shapes(test_ids, test_path)
    dataset_test.prepare()

    new_test_ids = []

    # No masks prediction counter
    no_masks = 0
    colour_model = joblib.load('./models/random_forest_model.pkl')
    for mm, image_id in tqdm(enumerate(dataset_test.image_ids)):
        # Load the image
        image = dataset_test.load_image(image_id, color=config.IMAGE_COLOR)
        # Image name for submission rows.
        image_id = dataset_test.image_info[image_id]['img_name']
        height, width = image.shape[:2]

        result = ensemble_prediction(model, config, image)
        #result = cluster_prediction(model, config, image)
        # result = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]

        # Clean overlaps and apply some post-processing
        result = postprocess_masks(result, image)
        # If there is no masks then try to predict on scaled image
        if result['masks'].sum() < 2:
            print("scaled image")
            H, W = image.shape[:2]
            scaled_img = np.zeros([4*H, 4*W, 3], np.uint8)
            scaled_img[:H, :W] = image
            result = cluster_prediction(model, config, scaled_img)
            result['masks'] = result['masks'][:H, :W]
            result = postprocess_masks(result, image)

        if result['masks'].sum() < 1:
            no_masks += 1

        class_col=[]
        l=[]
        tbd = []
        tbd_count=0
        for i in range(result['masks'].shape[2]):
            mask = result['masks'][:, :, i]
            indices = np.where(mask == 1)  # Get indices of cells with value 1
            if len(indices[0]) > 0:
                centroid_x = int(np.mean(indices[1]))  # Compute centroid x-coordinate
                centroid_y = int(np.mean(indices[0])) # Compute centroid y-coordinate
                p = [0,0,0]
                try:
                    for i in range(centroid_x-1,centroid_x+2):
                        for j in range(centroid_y-1,centroid_y+2):
                            p+= image[j,i]
                    p[0]/=9
                    p[1]/=9
                    p[2]/=9
                except:
                    p = image[centroid_y,centroid_x]
                p=rgb2lab(p)
                class_col.append(p)
                l.append(p[0])
            else:
                tbd.append(i)
        
        for i in tbd:
            result['masks'] = np.delete(result['masks'],i-tbd_count,2)
            tbd_count+=1
                
        class_col =  colour_model.predict(class_col)
        for i in range(len(class_col)):
            if class_col[i]=="Blue":
                class_col[i]=0
            else:
                if l[i]>70:
                    class_col[i]=0
                elif l[i]>50:
                    class_col[i]=1
                elif l[i]>29:
                    class_col[i]=2
                else:
                    class_col[i]=3
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        plot_boundary(image, true_masks=None, pred_masks=result['masks'],classes=class_col,
                        ax=fig.add_subplot(gs[0]))

        fig.savefig('./saved2/' +str(np.count_nonzero(class_col == 0))+"_"+ str(np.count_nonzero(class_col == 1))+"_"+ str(np.count_nonzero(class_col == 2))+"_"+ str(np.count_nonzero(class_col == 3)) +'.png', bbox_inches='tight')
        plt.close()


config = KaggleBowlConfig()
config.GPU_COUNT = 1
config.IMAGES_PER_GPU = 1
config.BATCH_SIZE = 1
config.display()


model = get_model(config, model_path='./models/kaggle_bowl.h5')
pred_n_plot_test(model, config, test_path="./saved")
