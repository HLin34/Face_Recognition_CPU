# CPU Face Recognition

A local face recognition demo for still images and a webcam. It uses OpenCV’s CPU backend to detect faces, extract embeddings, and match them against locally enrolled identities.

## Features

- Enroll or update an identity from a reference photo.
- Identify a face in a still image using cosine similarity.
- Recognize one person at a time from a webcam. A match must persist across consecutive frames before the label is displayed.
- Run offline after downloading the models. The app stores enrollment labels and embeddings locally and does not record webcam footage.

## How it works

**YuNet** detects the face and its landmarks. **SFace** uses those landmarks to align the face and generate a 128-dimensional embedding. The app compares the embedding with the enrolled templates and returns the best match if its cosine similarity meets the configured threshold.

## Setup

From the project directory, create a dedicated Conda environment and install the dependencies.

```sh
conda create -n face-recognition-cpu python=3.13 pip
conda activate face-recognition-cpu
python -m pip install -r requirements.txt
```

Create a `models/` directory and download the two ONNX files from OpenCV Zoo using the download button on each file page.

| Model | License |
| --- | --- |
| [YuNet — `face_detection_yunet_2023mar.onnx`](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | MIT |
| [SFace — `face_recognition_sface_2021dec.onnx`](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) | Apache-2.0 |

## Usage

Place photos you have permission to use in `local-data/`. Use photos with one clearly visible face.

```sh
# Enroll a person from a reference photo
python face_recognition.py enroll person01 local-data/reference.jpg

# Identify a person in another photo
python face_recognition.py identify local-data/probe.jpg

# Start live recognition
python face_recognition.py live --camera 0 --stable-frames 5
```

Press `q` or `Esc` to exit webcam mode. Use `--threshold` with `identify` or `live` to adjust the match threshold. The default of `0.363` comes from OpenCV’s LFW benchmark example. Suitable thresholds depend on your camera, lighting, and test data.

## Verification and limitations

Still-image enrollment and matching were tested on Windows. The webcam’s consecutive-frame logic was tested with simulated frames. Physical webcam operation has not been verified.

This demo does not perform liveness detection. Keep reference photos and face embeddings out of version control.
