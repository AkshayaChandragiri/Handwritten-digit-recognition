# main.py
# Handwritten Digit Recognition using CNN (MNIST)
# Supports image, webcam, and drawing GUI

import os
import argparse
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from keras import layers , models

MODEL_PATH = "mnist_cnn.h5"

# ----------------------------
# Preprocessing (MNIST-style)
# ----------------------------
def preprocess_for_model(img_path, debug_save="processed_preview.png"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Cannot read image")

    # Invert (black background, white digit)
    img = cv2.bitwise_not(img)

    # Threshold
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize digit to fit 20x20 box
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit = cv2.resize(digit, (new_w, new_h))

    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center the digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Normalize
    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)

    # Save debug image
    cv2.imwrite(debug_save, (canvas[0] * 255).astype(np.uint8))

    return canvas

# ----------------------------
# CNN Model
# ----------------------------
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=128,
    )
    model.save(MODEL_PATH)
    print("Model saved:", MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return train_model()

# ----------------------------
# Prediction
# ----------------------------
def predict_image(img_path, model):
    x = preprocess_for_model(img_path)
    probs = model.predict(x)
    pred = int(np.argmax(probs))
    print("Predicted digit:", pred)
    print("Processed image saved as processed_preview.png")

# ----------------------------
# Webcam
# ----------------------------
def webcam_predict(model):
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == 32:
            cv2.imwrite("capture.png", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(cv2.imread("capture.png"), cv2.COLOR_BGR2GRAY)
    cv2.imwrite("capture.png", gray)
    predict_image("capture.png", model)

# ----------------------------
# GUI
# ----------------------------
def run_gui(model):
    import tkinter as tk
    from tkinter import messagebox

    SIZE = 280
    root = tk.Tk()
    root.title("Draw Digit")

    canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="white")
    canvas.pack()

    drawing = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    last = [None]

    def draw(event):
        if last[0] is None:
            last[0] = (event.x, event.y)
            return
        x1, y1 = last[0]
        x2, y2 = event.x, event.y
        canvas.create_line(x1, y1, x2, y2, width=12, fill="black", capstyle="round")
        cv2.line(drawing, (x1, y1), (x2, y2), 0, 12)
        last[0] = (x2, y2)

    def clear():
        canvas.delete("all")
        drawing[:] = 255

    def predict():
        cv2.imwrite("gui.png", drawing)
        predict_image("gui.png", model)
        messagebox.showinfo("Result", "Check terminal output")

    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", lambda e: last.__setitem__(0, None))

    tk.Button(root, text="Predict", command=predict).pack(side="left")
    tk.Button(root, text="Clear", command=clear).pack(side="left")
    root.mainloop()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    model = load_model()

    if args.train:
        train_model()
    elif args.predict:
        predict_image(args.predict, model)
    elif args.webcam:
        webcam_predict(model)
    else:
        run_gui(model)

if __name__ == "__main__":
    main()
