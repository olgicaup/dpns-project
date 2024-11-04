import cv2
import numpy as np


def extract_descriptor(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Here I'm loading the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # GaussianBlur - to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the iris
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the region of interest (ROI) around the iris
        x, y, w, h = cv2.boundingRect(largest_contour)
        iris_roi = gray[y:y + h, x:x + w]

        # Resizing the iris region to generate a
        # descriptor to use it after
        resized_iris = cv2.resize(iris_roi, (128, 64))

        # With the following lines we'll extract color histogram
        # features and print them, then we will print
        # the color histogram values
        color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_hist = color_hist.flatten()

        print("Color Histogram Values:")
        for channel in range(3):
            channel_values = color_hist[channel * 64: (channel + 1) * 64]
            print(f"Channel {channel + 1}: {channel_values.sum()}")

        # Here we are combining iris texture
        # and color histogram features
        descriptor = np.concatenate((resized_iris.flatten(), color_hist))

        cv2.imshow('Cropped Iris Region', iris_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return descriptor
    else:
        print("No iris found in the image.")
        return None


def compare_irises(descriptor1, descriptor2):
    # Use Euclidean distance as
    # a similarity metric. You can set a threshold to determine
    # whether the irises are similar or not.
    # Return True if the distance is below the
    # threshold, indicating similarity.

    euclidean_distance = np.linalg.norm(descriptor1 - descriptor2)

    similarity_threshold = 50.0

    return euclidean_distance < similarity_threshold


image_path1 = 'images/iris-image2.jpg'
descriptor1 = extract_descriptor(image_path1)

image_path2 = 'images/iris-image3.jpg'
descriptor2 = extract_descriptor(image_path2)

# Here we are comparing the irises
are_similar = compare_irises(descriptor1, descriptor2)

if are_similar:
    print("The irises are similar.")
else:
    print("The irises are not similar.")
