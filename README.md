# Digit Recognition 

A simple browser-based app that recognizes handwritten digits (0–9). Draw a digit on an HTML5 canvas, and a pre-trained neural network (running fully client-side via **ONNX Runtime Web**) predicts what number you drew — no backend or server-side inference required.

## How it works

1. You draw a digit on a 280×280 canvas (white strokes on a black background, matching the MNIST format).
2. On clicking **Predict**, the app:
   - Finds the bounding box of your drawing.
   - Crops and rescales it into a centered 20×20 region inside a 28×28 frame (the classic MNIST preprocessing).
   - Normalizes pixel values using the same mean/std used during training (`0.1307` / `0.3081`).
   - Feeds the resulting 1×1×28×28 tensor into the ONNX model (`mnist_model.onnx`) using `onnxruntime-web`.
   - Applies a softmax to the model's output logits and displays the predicted digit with a confidence percentage.

## Tech stack

- **HTML5 Canvas** for drawing input
- **Vanilla JavaScript** (no framework, no build step)
- **[ONNX Runtime Web](https://github.com/microsoft/onnxruntime)** (loaded via CDN) to run the `.onnx` model directly in the browser
- **CSS** for styling

## Project structure

```
Digit-recognition/
├── index.html          # App markup, loads onnxruntime-web + index.js
├── index.js            # Canvas drawing, image preprocessing, inference logic
├── style.css            # Styling
├── mnist_model.onnx    # Pre-trained MNIST digit classifier (ONNX format)
└── README.md
```

## Running it on your computer

Since the app loads a local file (`mnist_model.onnx`) via `fetch`, you **can't** just double-click `index.html` and open it directly in the browser — most browsers block local file requests (`file://`) for security reasons. You need to serve the folder over `http://` using a simple local web server.

### 1. Clone the repository

```bash
git clone https://github.com/moohameedd/Digit-recognition.git
cd Digit-recognition
```

### 2. Start a local server

Pick whichever you have available:

**Option A — Python (built-in on most systems)**
```bash
python3 -m http.server 8000
```

**Option B — Node.js**
```bash
npx serve .
# or
npx http-server -p 8000
```

**Option C — VS Code**
Install the "Live Server" extension, right-click `index.html`, and choose **Open with Live Server**.

### 3. Open it in your browser

Navigate to:
```
http://localhost:8000
```

You should see "Model loading..." briefly, followed by "Model ready! Draw a digit and click Predict." Draw a digit with your mouse and click **Predict**.

## Requirements

- Any modern browser (Chrome, Firefox, Edge, Safari)
- An internet connection on first load (to fetch `onnxruntime-web` from the jsDelivr CDN)
- Python 3 or Node.js (only needed to serve the files locally — see above)

## Notes

- Drawing input uses the mouse only (`mousedown`/`mousemove`/`mouseup`); touch-screen support isn't implemented.
- The model expects a single centered digit — drawing multiple digits or very thin/off-center strokes may reduce accuracy.
- Click **Clear** to reset the canvas before drawing a new digit.