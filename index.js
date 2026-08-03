let session;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// 1. Initialize Canvas Background & Settings
function initCanvas() {
    // Fill background with solid black (MNIST format)
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set line style for drawing
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}
initCanvas();

// 2. Initialize ONNX Model
async function initModel() {
    try {
        console.log("Loading ONNX model...");
        session = await ort.InferenceSession.create('./mnist_model.onnx');
        console.log("ONNX Model loaded successfully!");
        document.getElementById('result').innerText = "Model ready! Draw a digit and click Predict.";
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        document.getElementById('result').innerText = "Error loading model. Check browser console (F12).";
    }
}
initModel();

// 3. Canvas Mouse Drawing Event Listeners
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);

// 4. Clear Canvas Function
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = "Draw a digit and click Predict!";
}

function getProcessedImageData() {
    // 1. Find bounding box of drawing on main canvas
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    let hasPixels = false;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const index = (y * canvas.width + x) * 4;
            if (imgData.data[index] > 20) { // white pixel threshold
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                hasPixels = true;
            }
        }
    }

    // If canvas is empty, return empty tensor
    if (!hasPixels) return new ort.Tensor('float32', new Float32Array(28 * 28), [1, 1, 28, 28]);

    // 2. Crop & center digit onto a temporary 28x28 canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill background black
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, 28, 28);

    const digitWidth = maxX - minX;
    const digitHeight = maxY - minY;

    // Scale digit to fit in a 20x20 box inside 28x28 (preserving aspect ratio)
    const scale = Math.min(20 / digitWidth, 20 / digitHeight);
    const sw = digitWidth * scale;
    const sh = digitHeight * scale;

    // Center in 28x28 frame
    const dx = (28 - sw) / 2;
    const dy = (28 - sh) / 2;

    tempCtx.drawImage(canvas, minX, minY, digitWidth, digitHeight, dx, dy, sw, sh);

    // 3. Extract final centered pixel tensor
    const finalImgData = tempCtx.getImageData(0, 0, 28, 28);
    const float32Data = new Float32Array(28 * 28);

    for (let i = 0; i < finalImgData.data.length; i += 4) {
        let rawPixel = finalImgData.data[i] / 255.0;
        
        // Match normalization used in training
        float32Data[i / 4] = (rawPixel - 0.1307) / 0.3081; 
    }

    return new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
}

// 7. Predict Function
async function predict() {
    if (!session) {
        alert("Model is still loading, please wait!");
        return;
    }

    const inputTensor = getProcessedImageData();
    const feeds = { input: inputTensor };

    const results = await session.run(feeds);
    const logits = Array.from(results.output.data);

    const probabilities = softmax(logits);
    const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
    const confidence = (probabilities[predictedDigit] * 100).toFixed(2);

    document.getElementById('result').innerText = `Prediction: ${predictedDigit} (${confidence}%)`;
}