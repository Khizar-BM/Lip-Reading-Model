import cv2
import os
import tensorflow as tf
from imutils import face_utils
import dlib
import numpy as np
import imageio

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def make_gif(frames, output_path):
    imageio.mimsave(output_path,
                    tf.image.convert_image_dtype(tf.image.resize(tf.squeeze(frames, -1), (600, 50)), dtype=tf.uint8),
                    fps=10)


def get_roi_with_markings(frame):
    faces = detector(frame)
    fixed_size = (100, 50)  # width x height
    marked_frame = frame.copy()  # Copy of the frame to draw markings
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    for face in faces:
        shape = predictor(frame, face)
        shape = face_utils.shape_to_np(shape)

        # Draw all facial landmarks
        for (x, y) in shape:
            cv2.circle(marked_frame, (x, y), 1, (0, 255, 0), -1)

        # Extract and draw rectangle around the mouth
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "mouth":
                # ... [existing code to calculate ROI] ...
                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                expansion_factor = 0.4  # e.g., 20% expansion
                new_w = int(w * (1 + expansion_factor))
                new_h = int(h * (1 + expansion_factor))

                # Adjust the top-left corner to compensate for the expansion
                new_x = max(x - int(w * expansion_factor / 2), 0)
                new_y = max(y - int(h * expansion_factor / 2), 0)

                # Ensure the expanded ROI doesn't exceed the frame boundaries
                new_x_end = min(new_x + new_w, frame.shape[1])
                new_y_end = min(new_y + new_h, frame.shape[0])

                cv2.rectangle(marked_frame, (new_x, new_y), (new_x_end, new_y_end), (0, 0, 255), 2)

                roi = frame[new_y:new_y_end, new_x:new_x_end]
                roi_resized = cv2.resize(roi, fixed_size)
                return roi_resized, marked_frame

    default_roi = frame[190:236, 80:220]
    return cv2.resize(default_roi, fixed_size), marked_frame


def load_video(path):
    cap = cv2.VideoCapture(path)
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(5)
    print(width, height, fps)
    processed_frames = []
    roi_frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        roi, marked_frame = get_roi_with_markings(frame)

        roi_frames.append(tf.expand_dims(tf.image.convert_image_dtype(roi, dtype=tf.uint8), -1))
        processed_frames.append(marked_frame)

    cap.release()

    mean = tf.math.reduce_mean(roi_frames)
    std = tf.math.reduce_std(tf.cast(roi_frames, tf.float32))
    return tf.cast((roi_frames - mean), tf.float32) / std, processed_frames


def load_alignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil':
                tokens = [*tokens, ' ', line[2]]
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path):
    path = path.numpy().decode('UTF-8')
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames, marked_frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, marked_frames, alignments
