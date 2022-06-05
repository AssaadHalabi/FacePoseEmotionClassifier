import os
import cv2
import numpy as np
import dlib
import imutils
from imutils.face_utils import FaceAligner
from threading import Lock

from typing import List, Tuple, Union

resume_lock = Lock()
NEW_IMAGE_SIZE = (224, 224)


def standardize_fname(fname: str) -> str:
    """
    Standardize file names to end with ".jpg".

    Some files are not jpgs, or are named weirdly like .JPG or .jpeg, so this fixes that.

    Arguments:
        fname (str): The file name to standardize.

    Returns:
        fname (str): The standardized file name.
    """
    fname = os.path.splitext(fname)[0]  # Strip file extension
    fname = fname + ".jpg"  # Set extension to .jpg
    return fname


def generate_save_path(path: str, output_dir: str, output_class: Union[int, None] = None) -> str:
    """
    Generate path to save image to.

    Files will be organized in directories by output class.
    e.g. a file of output class "0" will be saved in "dataset/output_dir/0/filename.jpg".

    Arguments:
        path (str): The original file path.
        output_dir (str): The root directory to store images in.
        output_class (int OR None): The subdirectory to store the image in, or directly
                                        in the root directory if not provided.

    Returns:
        new_path (str): The path to save the image to.
    """
    fname = os.path.basename(path)
    fname = standardize_fname(fname)
    if output_class is not None:
        new_path = os.path.join(
            output_dir,
            str(output_class),
            fname
        )
    else:
        new_path = os.path.join(
            output_dir,
            fname
        )
    return new_path


def preprocess_image_labelled(path: str, new_path: str, x: int, y: int, w: int, h: int) -> None:
    """
    Preprocess a labelled image.

    Expand the bounding box of the face by 10%, crop it, resize, and save it as a jpg.

    Arguments:
        path (str): The original file path.
        new_path (str): The file path to save the processed image to.
        x (int): The top left x coordinate of the bounding box of the face.
        y (int): The top left y coordinate of the bounding box of the face.
        w (int): The width of the bounding box of the face.
        h (int): The height of the bounding box of the face.
    """
    image = cv2.imread(path)

    # If image is empty, skip
    if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
        return

    scale_factor = 0.10

    length = w  # Is a square

    left = int(x - scale_factor * length)
    # Ensure doesn't overflow the bounds of the original image
    left = max(0, left)

    right = int(x + length + scale_factor * length)
    right = min(image.shape[1], right)

    top = int(y - scale_factor * length)
    top = max(0, top)

    bottom = int(y + length + scale_factor * length)
    bottom = min(image.shape[0], bottom)

    image = image[top:bottom, left:right]

    resized_image = cv2.resize(image, NEW_IMAGE_SIZE)
    cv2.imwrite(new_path, resized_image)


def preprocess_image_unlabelled(path: str, new_path: str) -> None:
    """
    Preprocess an unlabelled image.

    Resize, and save it as a jpg.

    Arguments:
        path (str): The original file path.
        new_path (str): The file path to save the processed image to.
    """
    image = cv2.imread(path)

    # If image is empty, skip
    if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
        return

    resized_image = cv2.resize(image, NEW_IMAGE_SIZE)
    cv2.imwrite(new_path, resized_image)


def preprocess_labelled_images_from_list(paths: List[str], new_paths: List[str], x_y_w_h: List[List[int]], resume_file: str) -> None:
    """
    Preprocess one labelled image at a time and mark image as processed in the resume file.
    """
    with open(resume_file, "a") as f:
        for path, new_path, (x, y, w, h) in zip(paths, new_paths, x_y_w_h):
            preprocess_image_labelled(path, new_path, x, y, w, h)
            resume_lock.acquire()
            f.write(path + "\n")
            f.flush()
            resume_lock.release()


def preprocess_unlabelled_images_from_list(paths: List[str], new_paths: List[str], resume_file: str) -> None:
    """
    Preprocess one unlabelled image at a time and mark image as processed in the resume file.
    """
    with open(resume_file, "a") as f:
        for path, new_path in zip(paths, new_paths):
            preprocess_image_unlabelled(path, new_path)
            resume_lock.acquire()
            f.write(path + "\n")
            f.flush()
            resume_lock.release()


def align_face(path: str, new_path: str, detector, fa) -> None:
    image = cv2.imread(path)
    # if image.shape[0] > 600 or image.shape[1] > 600:
    #     image = imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    for rect in rects[:1]:
        faceAligned = fa.align(image, gray, rect)
        cv2.imwrite(new_path, faceAligned)


def align_faces_from_list(paths: List[str], new_paths: List[str], resume_file: str) -> None:
    """
    Align the face of one image at a time and mark image as processed in the resume file.
    """

    with open(resume_file, "a") as f:
        for path, new_path in zip(paths, new_paths):
            align_face(path, new_path, detector, fa)
            resume_lock.acquire()
            f.write(path + "\n")
            f.flush()
            resume_lock.release()
