const { invoke } = window.__TAURI__.tauri;

import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import { OneEuroFilter } from "./lib/OneEuroFilter.js";
import { euclideanDistance, convertRange } from "./lib/helper.js"

let handLandmarker = undefined;
let enableWebcamButton;
let webcamRunning = false;

/**
 * avilable modes: DOWN, FRONT 
*/
const mode = "FRONT";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const init = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  
  if (mode == "FRONT") {
    video.style.transform = "scaleX(-1)";
    canvasElement.style.transform = "scaleX(-1)";
  } else if (mode == "DOWN") {
    video.style.transform = "scaleX(-1) scaleY(-1)";
    canvasElement.style.transform = "scaleX(-1) scaleY(-1)";
  }

  const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

  if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
};
init();

function enableCam() {
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

  const constraints = {
    video: true
  };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results = undefined;

let buffer = []

const PINCH_THRESHOLD = 9;
const BUFFER_SIZE = 9;
const PINCH_BUFFER_TOLERANCE = 0.5

const debugElement = document.getElementById("debug");

// freq: must be the same as how fast the data is inputted, for me this is 75
// under is more jitter
// upper is more lag
// mincutoff (start 1) -> lower to reduce jitter
// beta (start 0) -> increase beta to minimize lag
// dcutoff ?

const fx = new OneEuroFilter(60, 0.001, 0.01, 1);
const fy = new OneEuroFilter(60, 0.001, 0.01, 1);

const fxz = new OneEuroFilter(60, 0.001, 0.1, 1);
const fyz = new OneEuroFilter(60, 0.001, 0.1, 1);

const ANGLE_THRESHOLD = 0.02;
let prevXZAngle = 0;
let prevYZAngle = 0;

function xzAngleThreshold(xzAngle) {
  if (Math.abs(xzAngle - prevXZAngle) > ANGLE_THRESHOLD) {
    prevXZAngle = xzAngle;
  }
  return prevXZAngle;
}

function yzAngleThreshold(yzAngle) {
  if (Math.abs(yzAngle - prevYZAngle) > ANGLE_THRESHOLD) {
    prevYZAngle = yzAngle;
  }
  return prevYZAngle;
}

async function frontFacing(landmarks, timestamp) {
  // to get screen distance, we first need to calibrate, this is to take distance between thumbCmc and thumbMcp
  // (this is chosen for no particular reason, we can try different distance combos, as long as its length don't decrease)
  // then we just calculate the scale to be the distance unit
  // formula: distance unit = dist.calibration/dist.current

  // 1. determine pos
  const calibrationMax = 0.2;
  const calibrationMin = 0.04;
  const middlePip = landmarks[0][10];
  const middleMcp = landmarks[0][9];

  const distanceMiddlePipMiddleMcp = euclideanDistance(middlePip, middleMcp);

  const distToScreen = 0.4 * (1 - convertRange(distanceMiddlePipMiddleMcp, calibrationMin, calibrationMax, 0, 1));

  const deltaX = middleMcp.x - middlePip.x;
  const deltaY = middleMcp.y - middlePip.y;
  const deltaZ = middleMcp.z - middlePip.z;

  let angleXZ = fxz.filter(deltaX / deltaZ, timestamp);
  let angleYZ = fyz.filter(deltaY / deltaZ, timestamp);

  angleXZ = xzAngleThreshold(angleXZ);
  angleYZ = yzAngleThreshold(angleYZ);

  const pointerX = convertRange(1 - middleMcp.x + distToScreen * angleXZ, 0.1, 0.9, 0, 1);
  const pointerY = convertRange(middleMcp.y - distToScreen * angleYZ - 0.5, 0.2, 0.8, 0, 1);

  const inScreenX = pointerX >= 0 && pointerX <= 1;
  const inScreenY = pointerY >= 0 && pointerY <= 1;

  // 2. determine pinch
  const thumbTip = landmarks[0][4];
  const indexTip = landmarks[0][8];

  const thumbIp = landmarks[0][3];
  const indexDip = landmarks[0][7];

  const distanceThumbTipIndexTip = euclideanDistance(thumbTip, indexTip);
  const distanceThumbTipThumbIp = euclideanDistance(thumbTip, thumbIp);
  const distanceIndexTipIndexDip = euclideanDistance(indexTip, indexDip);

  const relativeDistance = (distanceThumbTipIndexTip * 10) / (0.5 * (distanceThumbTipThumbIp + distanceIndexTipIndexDip));

  let result = {
    x: fx.filter(pointerX, timestamp) * window.screen.width,
    y: fy.filter(pointerY, timestamp) * window.screen.height,
    pinch: relativeDistance <= PINCH_THRESHOLD
  }

  if (buffer.length < BUFFER_SIZE) {
    buffer.push({ ...result })
  } else {
    buffer.shift();
    buffer.push({ ...result })

    // result.x = buffer.map(b => b.x).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    // result.y = buffer.map(b => b.y).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    result.pinch = buffer.map(b => b.pinch).reduce((a, b) => a + b, 0) >= (PINCH_BUFFER_TOLERANCE * BUFFER_SIZE)
  }

  debugElement.innerText =
    `Middle Pip (x,y,z): ${Math.floor(middlePip.x * 1000)},${Math.floor(middlePip.y * 1000)},${Math.floor(middlePip.z * 1000)}
Middle Mcp (x,y,z): ${Math.floor(middleMcp.x * 1000)},${Math.floor(middleMcp.y * 1000)},${Math.floor(middleMcp.z * 1000)}
Dist Middle Pip - Middle Mcp: ${distanceMiddlePipMiddleMcp}
Screen Distance: ${distToScreen}
Delta X: ${Math.floor(deltaX * 1000)}
Delta Y: ${Math.floor(deltaY * 1000)}
Delta Z: ${Math.floor(deltaZ * 1000)}
AngleXZ: ${Math.atan(angleXZ) * (180 / Math.PI)}
AngleYZ: ${Math.atan(angleYZ) * (180 / Math.PI)}
Cursor Position: ${pointerX},${pointerY}
In Screen: ${inScreenX && inScreenY}
Cursor Position (screen): ${Math.floor(result.x)},${Math.floor(result.y)}
Relative Distance: ${relativeDistance}
Is Pinched: ${result.pinch}`;

  await invoke("mouse_action", { x: Math.floor(result.x), y: Math.floor(result.y), pinch: false });
}

async function downFacing(landmarks, timestamp) {
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

  result.x = fx.filter(result.x, timestamp);
  result.y = fy.filter(result.y, timestamp);

  if (buffer.length < BUFFER_SIZE) {
    buffer.push({ ...result })
  } else {
    buffer.shift();
    buffer.push({ ...result })

    // result.x = buffer.map(b => b.x).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
    // result.y = buffer.map(b => b.y).reduce((a, b) => a + b, 0) / BUFFER_SIZE;
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

    if (mode == "FRONT") {
      await frontFacing(results.landmarks, startTimeMs);
    } else if (mode == "DOWN") {
      await downFacing(results.landmarks, startTimeMs);
    }
  }
  canvasCtx.restore();

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
