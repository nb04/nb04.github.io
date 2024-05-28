const imageUpload = document.getElementById('imageUpload');
const imageCanvas = document.getElementById('imageCanvas');
const predictionResult = document.getElementById('predictionResult');
let model;

// Load the TensorFlow.js model
async function loadModel() {
    model = await tf.loadGraphModel('/tfjs-image-prediction/model/model.json'); // this line may need to be change
    console.log('Model loaded successfully');
}

loadModel();

// Handle image upload
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const image = new Image();
        image.src = URL.createObjectURL(file);
        image.onload = async () => {
            // Display the image on the canvas
            const ctx = imageCanvas.getContext('2d');
            imageCanvas.width = image.width;
            imageCanvas.height = image.height;
            ctx.drawImage(image, 0, 0);

            // Preprocess the image
            const tensor = tf.browser.fromPixels(image)
                .resizeNearestNeighbor([224, 224]) // Change to the required size
                .toFloat()
                .div(tf.scalar(255)) // Normalize the image to [0, 1] if required
                .expandDims();

            // Make a prediction
            const predictions = await model.predict(tensor).data();
            displayPrediction(predictions[0]);
        };
    }
});

// Display prediction result
function displayPrediction(prediction) {
    const classLabel = prediction > 0.5 ? 'Positive' : 'Negative'; // Adjust threshold as needed
    predictionResult.innerText = `Prediction: ${classLabel} with confidence ${(prediction * 100).toFixed(2)}%`;
}
