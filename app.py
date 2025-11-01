from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# ========== LOAD MODEL ==========
model = joblib.load("fruit_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# ========== IMAGE FEATURE EXTRACTION ==========
def extract_rgb_histogram(image, bins=8):
    img_uint8 = (image * 255).astype('uint8')
    rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)
    hist_r = cv2.calcHist([r], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [bins], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [bins], [0, 256])
    hist_r = cv2.normalize(hist_r, hist_r)
    hist_g = cv2.normalize(hist_g, hist_g)
    hist_b = cv2.normalize(hist_b, hist_b)
    return np.concatenate([hist_r, hist_g, hist_b]).flatten()

def extract_lbp_features(image):
    gray = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_shape_features(image):
    gray = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
    else:
        area, perimeter, aspect_ratio = 0, 0, 0
    return np.array([area / 10000.0, perimeter / 1000.0, aspect_ratio])

def extract_combined_features(image):
    color_hist = extract_rgb_histogram(image)
    texture = extract_lbp_features(image)
    shape = extract_shape_features(image)
    return np.hstack([color_hist, texture, shape])


# ========== ROUTES ==========
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return render_template('index.html', prediction="No file uploaded")

    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    # Extract features and predict
    features = extract_combined_features(img).reshape(1, -1)
    pred = model.predict(features)[0]
    predicted_label = le.inverse_transform([pred])[0]

    return render_template('index.html',
                           prediction=predicted_label,
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
