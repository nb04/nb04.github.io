const imageUpload = document.getElementById('imageUpload');
const imageCanvas = document.getElementById('imageCanvas');
const predictionResult = document.getElementById('predictionResult');
let model;


// Load the TensorFlow.js model
async function loadModel() {
    try {
        model = await tf.loadLayersModel('/tfjs-image-prediction/jsmodel_06_12/model.json'); // loadLayersModel instead of loadGraphModel
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
        image.onload = async () => {
            // Resize the image
            const resizedImage = await resizeImage(image);

            // Display the resized image on the canvas
            const ctx = imageCanvas.getContext('2d');
            imageCanvas.width = resizedImage.width;
            imageCanvas.height = resizedImage.height;
            ctx.drawImage(resizedImage, 0, 0);

            // Preprocess the image
            const tensor = tf.browser.fromPixels(resizedImage)
                .toFloat()

                // think this already happens in first layer of the model
                // .div(tf.scalar(255)) // Normalize the image to [0, 1] if required 
                
                .expandDims();

            // Make a prediction
            const predictions = await model.predict(tensor).data();
            displayPrediction(predictions[0]);
        };
    }
});

// Function to resize the image
async function resizeImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const targetWidth = 224; // Example target width (adjust as needed)
    const targetHeight = 224; // Example target height (adjust as needed)
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    return canvas;
}

// Display prediction result
function displayPrediction(prediction) {
    const classLabel = prediction > 0.5 ? 'Spoiled' : 'Fresh'; // Adjust threshold as needed
    predictionResult.innerText = `Prediction: ${classLabel} with confidence ${(prediction > 0.5 ? prediction * 100 : (1-prediction) * 100).toFixed(2)}%`;
}