let model;
let classIndices;

// Load the class indices
async function loadClassIndices() {
    const response = await fetch('class_indices.json');
    classIndices = await response.json();
}

// Function to get the class label by index
function getClassLabel(index) {
    return Object.keys(classIndices).find(key => classIndices[key] === index);
}

// Load the model
async function loadModel() {
    model = await tf.loadLayersModel('VGG_0.8acc_jsonModel/model.json');
    // Enable the predict button only after the model has been loaded and class indices are available
    await loadClassIndices();
    document.getElementById('predictButton').disabled = false;
}

loadModel();

// Handle image upload and prediction
document.getElementById('predictButton').addEventListener('click', async () => {
    const imageUpload = document.getElementById('imageUpload');
    if (imageUpload.files.length > 0) {
        const image = imageUpload.files[0];
        try {
            const prediction = await predict(image);
            // Get the class label from the prediction index
            const classLabel = getClassLabel(prediction);
            document.getElementById('prediction').innerText = `Prediction: ${classLabel}`;
        } catch (error) {
            console.error(error);
            document.getElementById('prediction').innerText = 'Error predicting image.';
        }
    } else {
        document.getElementById('prediction').innerText = 'Please upload an image first.';
    }
});

// Image prediction function
async function predict(image) {
    return new Promise((resolve, reject) => {
        // Read the image using the FileReader API
        const reader = new FileReader();
        reader.readAsDataURL(image);
        reader.onload = async (e) => {
            const img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                // Preprocess the image
                const processedImage = tf.tidy(() => {
                    let tensorImg = tf.browser.fromPixels(img)
                        .resizeNearestNeighbor([48, 48]) // Resize to the model's expected input size
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims(); // Add a batch dimension
                    return tensorImg;
                });
                // Make prediction
                const prediction = model.predict(processedImage);
                // Dispose the tensor to release memory
                processedImage.dispose();
                // Post-process prediction to display it
                const predictedIndex = prediction.argMax(1).dataSync()[0];
                prediction.dispose(); // Dispose the prediction tensor
                // Resolve the promise with the prediction
                resolve(predictedIndex);
            };
            img.onerror = () => {
                reject(new Error('Failed to load image.'));
            };
        };
        reader.onerror = () => {
            reject(new Error('Failed to read image file.'));
        };
    });
}