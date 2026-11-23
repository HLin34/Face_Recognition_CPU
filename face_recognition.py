"""Local, single-face recognition with OpenCV's CPU backend."""

import argparse
from pathlib import Path
import re
import tempfile

import cv2 as cv
import numpy as np


YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
VECTOR_SIZE = 128


class FaceEngine:
    def __init__(self, models):
        for filename in (YUNET, SFACE):
            if not (models / filename).is_file():
                raise ValueError(f"Missing model: {models / filename}. See README.md.")
        backend = cv.dnn.DNN_BACKEND_OPENCV
        target = cv.dnn.DNN_TARGET_CPU
        self.detector = cv.FaceDetectorYN.create(
            str(models / YUNET), "", (320, 320), 0.9, 0.3, 5000, backend, target
        )
        self.recognizer = cv.FaceRecognizerSF.create(
            str(models / SFACE), "", backend, target
        )

    def detect(self, image):
        height, width = image.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        if scale < 1:
            resized = cv.resize(
                image, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv.INTER_AREA,
            )
        else:
            resized = image
        resized_height, resized_width = resized.shape[:2]
        self.detector.setInputSize((resized_width, resized_height))
        _, faces = self.detector.detect(resized)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        faces[:, :14:2] *= width / resized_width
        faces[:, 1:14:2] *= height / resized_height
        return faces

    def embed(self, image, face):
        aligned = self.recognizer.alignCrop(image, face)
        return unit_vector(self.recognizer.feature(aligned).ravel())


def unit_vector(vector):
    if vector.shape != (VECTOR_SIZE,) or vector.dtype != np.float32:
        raise ValueError("Expected a 128-element SFace float32 vector.")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Face vector has an invalid norm.")
    return vector / norm


def read_image(path):
    image = cv.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def image_embedding(engine, path):
    image = read_image(path)
    faces = engine.detect(image)
    if len(faces) != 1:
        raise ValueError(f"Expected exactly one detected face, found {len(faces)}.")
    return engine.embed(image, faces[0])


def save_entry(database, label, vector):
    database.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=database, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("wb") as file:
            np.save(file, vector, allow_pickle=False)
        temporary.replace(database / f"id-{label}.npy")
    finally:
        temporary.unlink(missing_ok=True)


def load_database(database):
    entries = []
    for path in sorted(database.glob("id-*.npy")):
        label = path.stem.removeprefix("id-")
        if not LABEL.fullmatch(label):
            raise ValueError(f"Invalid database filename: {path}")
        try:
            entries.append((label, unit_vector(np.load(path, allow_pickle=False))))
        except (ValueError, TypeError, AttributeError, EOFError) as error:
            raise ValueError(f"Invalid SFace entry: {path}. Re-enroll this label.") from error
    if not entries:
        raise ValueError(f"No enrolled faces in {database}. Run enroll first.")
    return entries


def match(vector, entries, threshold):
    label, score = max(
        ((label, float(np.clip(vector @ reference, -1, 1))) for label, reference in entries),
        key=lambda item: item[1],
    )
    return (label if score >= threshold else None), score


def live(engine, entries, args):
    camera = cv.VideoCapture(args.camera)
    previous = None
    consecutive = 0
    try:
        if not camera.isOpened():
            raise ValueError(f"Cannot open camera {args.camera}.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise ValueError("Camera stopped returning frames.")
            faces = engine.detect(frame)
            label = None
            message = "No face" if len(faces) == 0 else "One person at a time"
            color = (0, 165, 255)
            if len(faces) == 1:
                face = faces[0]
                label, score = match(engine.embed(frame, face), entries, args.threshold)
                consecutive = consecutive + 1 if label and label == previous else int(bool(label))
                message = f"Unknown ({score:.2f})"
                if label:
                    message = "Confirming..."
                    if consecutive >= args.stable_frames:
                        message = f"{label} ({score:.2f})"
                        color = (0, 255, 0)
                x, y, width, height = map(int, face[:4])
                cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            else:
                consecutive = 0
            previous = label
            cv.putText(frame, message, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv.imshow("CPU Face Recognition", frame)
            if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv.getWindowProperty("CPU Face Recognition", cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--database", type=Path, default=Path("local-data/sface-v1"))
    commands = parser.add_subparsers(dest="command", required=True)
    enroll = commands.add_parser("enroll", help="Save or replace one local face template.")
    enroll.add_argument("label")
    enroll.add_argument("image", type=Path)
    identify = commands.add_parser("identify", help="Identify one face in a still image.")
    identify.add_argument("image", type=Path)
    webcam = commands.add_parser("live", help="Identify one face at a time from a webcam.")
    webcam.add_argument("--camera", type=int, default=0)
    webcam.add_argument("--stable-frames", type=int, default=5)
    for command in (identify, webcam):
        command.add_argument("--threshold", type=float, default=0.363)
    args = parser.parse_args()
    if args.command == "enroll" and not LABEL.fullmatch(args.label):
        parser.error("label must be 1–40 ASCII letters, digits, hyphens or underscores and start with a letter or digit")
    if args.command != "enroll" and not -1 <= args.threshold <= 1:
        parser.error("--threshold must be between -1 and 1")
    if args.command == "live" and (args.stable_frames < 1 or args.camera < 0):
        parser.error("--stable-frames must be positive and --camera must be nonnegative")
    try:
        engine = FaceEngine(args.models)
        if args.command == "enroll":
            save_entry(args.database, args.label, image_embedding(engine, args.image))
            print(f"Enrolled: {args.label}")
        else:
            entries = load_database(args.database)
            if args.command == "identify":
                label, score = match(image_embedding(engine, args.image), entries, args.threshold)
                print(f"{label or 'Unknown'} ({score:.4f})")
            else:
                live(engine, entries, args)
    except (OSError, ValueError, cv.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
