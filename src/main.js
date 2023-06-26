const { invoke } = window.__TAURI__.tauri;

import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let enableWebcamButton;
let webcamRunning = false;

/**
 * avilable modes: NORMAL, TOPDOWN
*/
const mode = "NORMAL";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const init = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  demosSection.classList.remove("invisible");
  if (mode == "NORMAL") {
    video.style.transform = "scaleX(-1)";
    canvasElement.style.transform = "scaleX(-1)";
  } else if (mode == "TOPDOWN") {
    video.style.transform = "scaleX(-1) scaleY(-1)";
    canvasElement.style.transform = "scaleX(-1) scaleY(-1)";
  }
};
init();

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results = undefined;

function euclideanDistance(point1, point2) {
  const deltaX = point2.x - point1.x;
  const deltaY = point2.y - point1.y;
  const deltaZ = point2.z - point1.z;

  const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2);
  return distance;
}

function convertRange(value, oldMin, oldMax, newMin, newMax) {
  const result = (value - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin;
  if (result < newMin) return newMin;
  if (result > newMax) return newMax;

  return result;
}

let buffer = []

const PINCH_THRESHOLD = 12;
const BUFFER_SIZE = 15;
const PINCH_BUFFER_TOLERANCE = 0.5

const debugElement = document.getElementById("debug");

async function normal(landmarks) {
  // to get screen distance, we first need to calibrate, this is to take distance between thumbCmc and thumbMcp
  // (this is chosen for no particular reason, we can try different distance combos, as long as its length don't decrease)
  // then we just calculate the scale to be the distance unit
  // formula: distance unit = dist.calibration/dist.current
  const distCalibration = 0.15;

  const thumbTip = landmarks[0][4];
  const thumbMcp = landmarks[0][2];
  const thumbCmc = landmarks[0][1];

  const distThumbMcpThumbCmc = euclideanDistance(thumbMcp, thumbCmc);
  const distToScreen = Math.max((distCalibration / distThumbMcpThumbCmc) - distCalibration, 0) / 20;

  const deltaX = thumbMcp.x - thumbTip.x;
  const deltaY = thumbMcp.y - thumbTip.y;
  const deltaZ = thumbMcp.z - thumbTip.z;

  const angleXZ = deltaX / deltaZ;
  const angleYZ = deltaY / deltaZ;

  const pointerX = 1 - thumbMcp.x + distToScreen * angleXZ;
  const pointerY = thumbMcp.y - distToScreen * angleYZ;

  const inScreenX = pointerX >= 0 && pointerX <= 1;
  const inScreenY = pointerY >= 0 && pointerY <= 1;

  let result = {
    x: pointerX * window.screen.width,
    y: pointerY * window.screen.height,
    pinch: false
  }

  if (buffer.length < BUFFER_SIZE) {
    buffer.push({ ...result })
  } else {
    buffer.shift();
    buffer.push({ ...result })

    result.x = buffer.map(b => b.x).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    result.y = buffer.map(b => b.y).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    result.pinch = buffer.map(b => b.pinch).reduce((a, b) => a + b, 0) >= (PINCH_BUFFER_TOLERANCE * BUFFER_SIZE)
  }

  debugElement.innerText =
    `Thumb Tip (x,y,z): ${Math.floor(thumbTip.x * 1000)},${Math.floor(thumbTip.y * 1000)},${Math.floor(thumbTip.z * 1000)}
Thumb Mcp (x,y,z): ${Math.floor(thumbMcp.x * 1000)},${Math.floor(thumbMcp.y * 1000)},${Math.floor(thumbMcp.z * 1000)}
Dist Thumb Mcp - Thumb Cmc: ${distThumbMcpThumbCmc}
Screen Distance: ${distToScreen}
Delta X: ${Math.floor(deltaX * 1000)}
Delta Y: ${Math.floor(deltaY * 1000)}
Delta Z: ${Math.floor(deltaZ * 1000)}
AngleXZ: ${angleXZ * (180 / Math.PI)}
AngleYZ: ${angleYZ * (180 / Math.PI)}
Cursor Position: ${pointerX},${pointerY}
In Screen: ${inScreenX && inScreenY}
Cursor Position (screen): ${Math.floor(result.x)},${Math.floor(result.y)}`;

  await invoke("mouse_action", { x: Math.floor(result.x), y: Math.floor(result.y), pinch: result.pinch });
}

async function topDown(landmarks) {
  const thumbTip = landmarks[0][4];
  const indexTip = landmarks[0][8];

  const thumbIp = landmarks[0][3];
  const indexDip = landmarks[0][7];

  const distanceThumbTipIndexTip = euclideanDistance(thumbTip, indexTip);
  const distanceThumbTipThumbIp = euclideanDistance(thumbTip, thumbIp);
  const distanceIndexTipIndexDip = euclideanDistance(indexTip, indexDip);

  const relativeDistance = (distanceThumbTipIndexTip * 10) / (0.5 * (distanceThumbTipThumbIp + distanceIndexTipIndexDip));
  const pointerX = convertRange(1 - thumbTip.x, 0.2, 0.8, 0, 1);
  const pointerY = convertRange(1 - thumbTip.y, 0.2, 0.8, 0, 1);

  let result = {
    x: pointerX * window.screen.width,
    y: pointerY * window.screen.height,
    pinch: relativeDistance <= PINCH_THRESHOLD
  }

  if (buffer.length < BUFFER_SIZE) {
    buffer.push({ ...result })
  } else {
    buffer.shift();
    buffer.push({ ...result })

    result.x = buffer.map(b => b.x).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    result.y = buffer.map(b => b.y).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    result.pinch = buffer.map(b => b.pinch).reduce((a, b) => a + b, 0) >= (PINCH_BUFFER_TOLERANCE * BUFFER_SIZE)
  }

  debugElement.innerText =
    `Relative Distance: ${relativeDistance}
Is Pinched: ${result.pinch}
Cursor Position: ${thumbTip.x},${thumbTip.y}
Cursor Position (screen): ${Math.floor(result.x)},${Math.floor(result.y)}`;

  await invoke("mouse_action", { x: Math.floor(result.x), y: Math.floor(result.y), pinch: result.pinch });
}

async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  // Now let's start detecting the stream.
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results && results.landmarks && results.landmarks.length > 0) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 3
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 1 });
    }

    if (mode == "NORMAL") {
      await normal(results.landmarks);
    } else if (mode == "TOPDOWN") {
      await topDown(results.landmarks);
    }
  }
  canvasCtx.restore();

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
