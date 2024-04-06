import streamlit as st
import cv2
import numpy as np
from PIL import Image
from joblib import load
# from tensorflow import keras
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from PIL import Image

def preprocess_image(image):
    # Preprocessing using Contrast Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(image)
    return equalized

def segment_image(image):
    # Segmentation using Otsu's Threshold Algorithm
    _, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented

def remove_brain_contour(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (presumably the brain contour)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask
    mask = np.zeros_like(segmented_image)

    # Draw all contours except the largest one (brain contour) on the mask
    for contour in contours:
        if contour is not largest_contour:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Invert the mask to retain the tumor area
    cleaned_image = cv2.bitwise_and(segmented_image, cv2.bitwise_not(mask))

    return cleaned_image, largest_contour

def remove_brain_margins(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(segmented_image)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Fill the contours to remove the exterior
    filled_mask = cv2.fillPoly(mask.copy(), contours, 255)

    # Invert the mask to retain the tumor area
    cleaned_image = cv2.bitwise_and(segmented_image, cv2.bitwise_not(filled_mask))
    return cleaned_image

def create_heatmap(image):
    # Convert the grayscale image to BGR format
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create a binary mask for the image
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply a color gradient to the mask
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

    return heatmap

def display_images(images, titles):
    num_images = len(images)
    num_rows = (num_images + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5*num_rows))

    for idx, ax in enumerate(axes.flatten()):
        if idx < num_images:
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(titles[idx])
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def find_num_connected_shapes(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary format
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # Apply connected component labeling
    num_labels, labeled_image = cv2.connectedComponents(binary_image)

    return num_labels - 1  # Subtract 1 to exclude the background label

def find_areas_of_connected_shapes(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary format
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # Apply connected component labeling
    num_labels, labeled_image = cv2.connectedComponents(binary_image)

    shape_areas = {}
    small_shapes = []

    for label in range(1, num_labels):
        # Create a mask for the current label
        mask = (labeled_image == label).astype(np.uint8)

        area = cv2.countNonZero(mask)
        shape_areas[label] = area
        if area < 10:
            small_shapes.append(label)

    return shape_areas, labeled_image, small_shapes

def remove_shapes(image, labeled_image, largest_label, small_shapes):
    # Create a mask to exclude the largest shape and small shapes
    mask = np.ones_like(labeled_image, dtype=np.uint8)
    mask[(labeled_image == largest_label)] = 0
    for label in small_shapes:
        mask[(labeled_image == label)] = 0

    # Apply the mask to the original image
    result_image = image.copy()
    result_image[mask == 0] = 0

    return result_image

def find_bounding_box(labeled_image, largest_label):
    # Find the contours of the largest shape
    contours, _ = cv2.findContours((labeled_image == largest_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest shape
    x, y, w, h = cv2.boundingRect(contours[0])
    
    return x, y, w, h

def remove_shapes_outside_region(image, largest_contour):
    # Create a blank mask with the same dimensions as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw the contour of the largest region on the mask
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to remove shapes outside the region of interest
    result_image = cv2.bitwise_and(image, image, mask=mask)
    
    return result_image


sys.path.append(os.path.abspath(os.path.join('..')))


def get_tumorous_zone(image):
    # image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    print("Size of decoded image:", image.shape)  # Debug statement

    # Preprocess the image using Contrast Adaptive Histogram Equalization
    preprocessed_image = preprocess_image(image)

    # Preprocess the image using Contrast Adaptive Histogram Equalization
    preprocessed_image = preprocess_image(image)

    # Segment the preprocessed image using Otsu's Threshold Algorithm
    segmented_image = segment_image(preprocessed_image)
    cleaned_contour_image, largest_contour = remove_brain_contour(segmented_image)
    # cleaned_image = remove_brain_margins(cleaned_contour_image)
    heatmap = create_heatmap(image)

    num_shapes = find_num_connected_shapes(heatmap)
    print("Number of connected shapes:", num_shapes)

    areas, labeled_image, small_shapes = find_areas_of_connected_shapes(heatmap)
    largest_label = max(areas, key=areas.get)
    print("Areas of connected shapes:", areas)
    print("Small shapes to be removed:", small_shapes)

    x, y, w, h = find_bounding_box(labeled_image, largest_label)
    result_image = remove_shapes_outside_region(heatmap, largest_contour)

    # # Remove the largest shape and small shapes from the heatmap
    result_image = remove_shapes(result_image, labeled_image, largest_label, small_shapes)

    # cv2.imshow('Result Image!!!!', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result_image_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Find contours in the result_image
    contours, _ = cv2.findContours(result_image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw rectangles on
    image_with_rectangles = image.copy()

    # Iterate over each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Change the color of the rectangle (blue in this example)
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (255, 0, 0), 1)

    return image_with_rectangles

def preprocess_image_before_prediction(image):
    img = cv2.resize(image, (64, 64))
    # img_array = np.array(img)
    img_array = img.reshape(1, -1)  # Flatten the image to match expected shape
    img_array = img_array / 255.0
    return img_array

model_filename = "../xgboost_model.json"
model = xgb.XGBClassifier()
model.load_model(model_filename)
# model = keras.models.load_model('../brain_tumor_detection_model.h5')


def main():
    st.title('CT SCAN')

    uploaded_file = st.file_uploader('Upload your CT scan', type=["png","jpg","jpeg"])
    
    if uploaded_file is not None:
        
        file_bytes = np.asanyarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(original_image, caption='Uploaded CT Scan', use_column_width = True)
        
        input_image = preprocess_image_before_prediction(img)
        prediction = model.predict(input_image)

        st.write('Prediction:', prediction)

        if prediction == 1:
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            image_with_rectangles = get_tumorous_zone(image)
            st.image(image_with_rectangles, caption='Tumorous Zone Heatmap', use_column_width=True)



if __name__ == "__main__":
    main()
