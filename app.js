const imageUpload = document.getElementById('imageUpload');
const imageCanvas = document.getElementById('imageCanvas');
const predictionResult = document.getElementById('predictionResult');
const centerXOffset = document.getElementById('centerXOffset');
const centerYOffset = document.getElementById('centerYOffset');
const cropSize = document.getElementById('cropSize');
const centerXOffsetValue = document.getElementById('centerXOffsetValue');
const centerYOffsetValue = document.getElementById('centerYOffsetValue');
const cropSizeValue = document.getElementById('cropSizeValue');
const predictButton = document.getElementById('predictButton');
let model;
let uploadedImage;

// Load the TensorFlow.js model
async function loadModel() {
    try {
        model = await tf.loadLayersModel('/tfjs-image-prediction/jsmodel_07_15/model.json'); // loadLayersModel instead of loadGraphModel
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

loadModel();

// Handle image upload
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const image = new Image();
        image.src = URL.createObjectURL(file);
        image.onload = () => {
            uploadedImage = image;
            updateImageCanvas();
        };
    }
});

// Update the canvas and display the resized image
function updateImageCanvas() {
    if (!uploadedImage) return;

    const resizedImage = resizeImage(uploadedImage);
    const ctx = imageCanvas.getContext('2d');
    imageCanvas.width = resizedImage.width;
    imageCanvas.height = resizedImage.height;
    ctx.drawImage(resizedImage, 0, 0);
}

// Function to resize the image with cropping parameters
function resizeImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const cropWidth = parseInt(cropSize.value);
    const cropHeight = cropWidth;
    const centerX = image.width / 2 + parseInt(centerXOffset.value);
    const centerY = image.height / 2 + parseInt(centerYOffset.value);
    const startX = Math.max(0, centerX - cropWidth / 2);
    const startY = Math.max(0, centerY - cropHeight / 2);
    canvas.width = cropWidth;
    canvas.height = cropHeight;
    ctx.drawImage(image, startX, startY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

    // Resize to 224x224 pixels
    const resizedCanvas = document.createElement('canvas');
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCanvas.width = 224;
    resizedCanvas.height = 224;
    resizedCtx.drawImage(canvas, 0, 0, 224, 224);
    return resizedCanvas;
}

// Display prediction result
function displayPrediction(prediction) {
    const classLabel = prediction > 0.5 ? 'Spoiled' : 'Fresh'; // Adjust threshold as needed
    predictionResult.innerText = `Prediction: ${classLabel} with confidence ${(prediction > 0.5 ? prediction * 100 : (1 - prediction) * 100).toFixed(2)}%`;
}

// Handle prediction generation
predictButton.addEventListener('click', async () => {
    if (!uploadedImage) return;

    // Resize the image with the current cropping parameters
    const resizedImage = resizeImage(uploadedImage);

    // Preprocess the image
    const tensor = tf.browser.fromPixels(resizedImage)
        .toFloat()
        .div(tf.scalar(255.0)) // Normalize to [0, 1]
        .expandDims();

    // Make a prediction
    const predictions = await model.predict(tensor).data();
    displayPrediction(predictions[0]);
});

// Update slider values display
centerXOffset.addEventListener('input', () => {
    centerXOffsetValue.innerText = centerXOffset.value;
    updateImageCanvas();
});

centerYOffset.addEventListener('input', () => {
    centerYOffsetValue.innerText = centerYOffset.value;
    updateImageCanvas();
});

cropSize.addEventListener('input', () => {
    cropSizeValue.innerText = cropSize.value;
    updateImageCanvas();
});
