const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const CLASS_NAMES = [];
let mobilenet;
let model;
let gatherDataState = -1;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;
let webcamStream;
let capturing = false;
let captureInterval;
let previewing = false;
let previewInterval = null;

// Start webcam
async function startCamera() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('camera').srcObject = webcamStream;
    } catch (error) {
        alert('웹캠을 사용할 수 없습니다. 권한을 확인해주세요.');
    }
}

// Load MobileNet Feature Model
async function loadMobileNetFeatureModel() {
    try {
        const URL = 'static/model.json';
        let tmp_model = await tf.loadLayersModel(URL);
        const layer = tmp_model.getLayer('global_average_pooling2d_1');
        mobilenet = tf.model({inputs: tmp_model.inputs, outputs: layer.output});

        // Warm up the model by passing zeros through it once.
        tf.tidy(function () {
            mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        });
        document.getElementById('training-progress').innerText = '모델이 준비되었습니다.';
    } catch (error) {
        alert('MobileNet 모델을 로드하는 중 오류가 발생했습니다. 경로를 확인해주세요.');
    }
}

loadMobileNetFeatureModel();

// Add a new class
function addClass() {
    const className = document.getElementById('class-name').value.trim();
    if (className && !CLASS_NAMES.includes(className)) {
        CLASS_NAMES.push(className);
        // Create new class container
        const classContainer = document.createElement('div');
        classContainer.className = 'class-container';
        classContainer.id = `class-${className}`;
        classContainer.innerHTML = `<h3 onclick="selectClass('${className}')">${className}</h3><div class="image-collection"></div>`;
        classContainer.onclick = () => {
            selectClass(className);
        };
        // Add download button for each class
        const downloadButton = document.createElement('button');
        downloadButton.innerText = `데이터셋 다운로드`;
        downloadButton.onclick = (e) => {
            e.stopPropagation();
            downloadClassDataset(className);
        };
        classContainer.appendChild(downloadButton);
        // Add upload button for each class
        const uploadButton = document.createElement('input');
        uploadButton.type = 'file';
        uploadButton.accept = '.zip';
        uploadButton.style.display = 'none';
        uploadButton.onchange = (e) => {
            uploadClassDataset(e, className);
        };
        const uploadButtonLabel = document.createElement('label');
        uploadButtonLabel.innerText = `데이터셋 업로드`;
        uploadButtonLabel.classList.add('upload-button');
        uploadButtonLabel.onclick = () => {
            uploadButton.click();
        };
        classContainer.appendChild(uploadButton);
        classContainer.appendChild(uploadButtonLabel);
        document.getElementById('class-list').appendChild(classContainer);
        document.getElementById('class-name').value = '';
        selectClass(className); // 자동으로 새로 추가된 클래스를 선택
    } else {
        alert('유효한 클래스 이름을 입력하거나 중복되지 않은 이름을 사용하세요.');
    }
}

// Select a class
function selectClass(className) {
    // Clear previous selection
    document.querySelectorAll('.class-container').forEach(container => {
        container.classList.remove('selected');
    });

    // Update gatherDataState and highlight selected class
    gatherDataState = CLASS_NAMES.indexOf(className);
    const selectedClassContainer = document.getElementById(`class-${className}`);
    if (selectedClassContainer) {
        selectedClassContainer.classList.add('selected');
    }
}

// Capture images when button is held down
function startCapturingImages() {
    capturing = true;
    captureInterval = setInterval(() => {
        if (capturing) {
            captureImage();
        }
    }, 200); // Capture image every 200ms
}

function stopCapturingImages() {
    capturing = false;
    clearInterval(captureInterval);
}

// Capture image from webcam
function captureImage() {
    if (gatherDataState === -1) {
        alert('이미지를 추가할 클래스를 선택하세요.');
        return;
    }

    const video = document.getElementById('camera');
    const imgTensor = tf.tidy(() => {
        let img = tf.browser.fromPixels(video)
            .resizeBilinear([MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH])
            .toFloat()
            .div(tf.scalar(255));
        return tf.image.per_image_standardization(img);
    });

    addImageToClass(imgTensor, gatherDataState);
}

// Helper function to add image tensor to class and feature extraction
function addImageToClass(imgTensor, classIndex) {
    const features = mobilenet.predict(imgTensor.expandDims()).squeeze();
    trainingDataInputs.push(features);
    trainingDataOutputs.push(classIndex);
    examplesCount[classIndex] = (examplesCount[classIndex] || 0) + 1;
    updateStatus();
}

// Update status message
function updateStatus() {
    const status = document.getElementById('training-progress');
    status.innerText = CLASS_NAMES.map((name, index) => `${name} data count: ${examplesCount[index] || 0}`).join('. ');
}

// Train model using MobileNet for transfer learning
async function trainAndPredict() {
    if (trainingDataInputs.length === 0) {
        alert('학습할 데이터가 없습니다. 이미지를 추가해주세요.');
        return;
    }

    predict = false;
    document.getElementById('training-progress').innerText = 'Training ...';
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1280], units: 128, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.5})); // 과적합 방지
    model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(inputsAsTensor, oneHotOutputs, {
        shuffle: true,
        batchSize: parseInt(document.getElementById('batch-size').value),
        epochs: parseInt(document.getElementById('epochs').value),
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log('Data for epoch ' + epoch, logs);
                document.getElementById('training-progress').innerText = `에포크 ${epoch + 1}: 손실 ${logs.loss.toFixed(4)}, 정확도 ${(logs.acc * 100).toFixed(2)}%`;
            }
        }
    }).then(() => {
        outputsAsTensor.dispose();
        oneHotOutputs.dispose();
        inputsAsTensor.dispose();
        document.getElementById('training-progress').innerText = '학습이 완료되었습니다.';
        predict = true;
        togglePredictImage();
    }).catch(error => {
        alert('학습 도중 오류가 발생했습니다: ' + error.message);
    });
}

// Toggle prediction preview
function togglePredictImage() {
    const previewStatus = document.getElementById('preview-status');
    if (!previewStatus) return;
    if (previewing) {
        clearInterval(previewInterval);
        previewing = false;
        previewStatus.style.visibility = 'hidden';
        document.getElementById('prediction-result').innerText = '아직 예측이 없습니다';
    } else {
        previewInterval = setInterval(() => predictImage(), 1000); // Predict every second
        previewing = true;
        previewStatus.innerText = '(미리보기 실행 중)';
        previewStatus.style.visibility = 'visible';
    }
}

// Predict using trained model
async function predictImage() {
    if (!model) {
        alert('먼저 모델을 학습하세요');
        return;
    }

    const video = document.getElementById('camera');
    const img = tf.tidy(() => tf.browser.fromPixels(video)
                    .resizeBilinear([MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH])
                    .toFloat()
                    .div(tf.scalar(255.0))
                    .expandDims());

    const features = tf.tidy(() => mobilenet.predict(img).flatten());
    const prediction = tf.tidy(() => model.predict(features.expandDims()));
    const predictionData = await prediction.data();
    const classIndex = prediction.argMax(-1).dataSync()[0];
    const confidence = predictionData[classIndex];

    const className = CLASS_NAMES[classIndex];
    document.getElementById('prediction-result').innerText = `예측된 클래스: ${className} (신뢰도: ${(confidence * 100).toFixed(2)}%)`;

    img.dispose();
    features.dispose();
    prediction.dispose();
}

// Upload dataset from ZIP (placeholder function)
function uploadClassDataset(event, className) {
    alert(`데이터셋 업로드 기능은 아직 구현되지 않았습니다. 클래스: ${className}`);
}

// Start the camera when the page loads
window.onload = () => {
    startCamera();
};
