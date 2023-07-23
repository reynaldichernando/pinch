const { invoke } = window.__TAURI__.tauri;

import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import { OneEuroFilter } from "./lib/OneEuroFilter.js";
import { euclideanDistance, convertRange } from "./lib/helper.js"

let handLandmarker = undefined;
let webcamRunning = false;
let cursorEnabled = false;

const toggleCursorElement = document.getElementById("toggleCursor");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const debugElement = document.getElementById("debug");

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

  video.style.transform = "scaleX(-1)";
  canvasElement.style.transform = "scaleX(-1)";

  toggleCursorElement.addEventListener('click', () => {
    cursorEnabled = !cursorEnabled;
    if (cursorEnabled) {
      toggleCursorElement.innerHTML = 'Disable'
    } else {
      toggleCursorElement.innerHTML = 'Enable&nbsp;'
    }
  })

  const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

  if (hasGetUserMedia()) {
    webcamRunning = true;
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predict);
    });
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
};
init();

let lastVideoTime = -1;
let results = undefined;

let buffer = []

const PINCH_DISTANCE_THRESHOLD = 12;
const PINCH_BUFFER_SIZE = 11;
const PINCH_BUFFER_TOLERANCE = 0.5

// freq: camera fps
// mincutoff (start with 1): lower to reduce jitter
// beta (start with 0): increase beta to minimize lag
// dcutoff: keep as default

const fx = new OneEuroFilter(60, 0.001, 0.1, 1);
const fy = new OneEuroFilter(60, 0.001, 0.1, 1);

const fxz = new OneEuroFilter(60, 0.001, 0.1, 1);
const fyz = new OneEuroFilter(60, 0.001, 0.1, 1);

const ANGLE_THRESHOLD = 0.03;
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

let delayCounter = 0;

async function driver(landmarks, timestamp) {
  // determine pinch
  const thumbTip = landmarks[0][4];
  const indexTip = landmarks[0][8];

  const thumbIp = landmarks[0][3];
  const indexDip = landmarks[0][7];

  const distanceThumbTipIndexTip = euclideanDistance(thumbTip, indexTip);
  const distanceThumbTipThumbIp = euclideanDistance(thumbTip, thumbIp);
  const distanceIndexTipIndexDip = euclideanDistance(indexTip, indexDip);

  const relativeDistance = (distanceThumbTipIndexTip * 10) / (0.5 * (distanceThumbTipThumbIp + distanceIndexTipIndexDip));

  let pinch = relativeDistance <= PINCH_DISTANCE_THRESHOLD;

  if (buffer.length < PINCH_BUFFER_SIZE) {
    buffer.push(pinch)
  } else {
    buffer.shift();
    buffer.push(pinch)

    pinch = buffer.reduce((a, b) => a + b, 0) >= (PINCH_BUFFER_TOLERANCE * PINCH_BUFFER_SIZE)
  }

  // to get screen distance, we first need to calibrate, this is to take distance between two finger points
  // (this is chosen for no particular reason, we can try different distance combos, as long as its length don't decrease)
  // then we just calculate the scale to be the distance unit

  // determine pos
  const calibrationMax = 0.2; // closest to the screen, this is using middlePip and middleMcp
  const calibrationMin = calibrationMax / 5; // magic number here, setting the calibration min to be the smallest value the distance could be

  const middlePip = landmarks[0][10];
  const middleMcp = landmarks[0][9];

  const distanceMiddlePipMiddleMcp = euclideanDistance(middlePip, middleMcp);

  // converting the value to 0-1 to make the pointer calculation works 
  // (pointer is value ranging from 0-1 which will later converted to screen coordinates)
  // inverting the value because, larger = closer to screen, smaller = further from screen

  const distToScreen = 1 - convertRange(distanceMiddlePipMiddleMcp, calibrationMin, calibrationMax, 0, 1);

  const deltaX = middleMcp.x - middlePip.x;
  const deltaY = middleMcp.y - middlePip.y;
  const deltaZ = middleMcp.z - middlePip.z;

  let angleXZ = fxz.filter(deltaX / deltaZ, timestamp);
  let angleYZ = fyz.filter(deltaY / deltaZ, timestamp);

  if (relativeDistance < PINCH_DISTANCE_THRESHOLD + 3 && !pinch) {
    // angle no change zone, to prevent drifting
    angleXZ = prevXZAngle;
    angleYZ = prevYZAngle;
    delayCounter = 15; // delay after pinch, for it to return using normal result
  } else {
    if (delayCounter > 0) {
      delayCounter -= 1;
      angleXZ = prevXZAngle;
      angleYZ = prevYZAngle;
    } else {
      angleXZ = xzAngleThreshold(angleXZ);
      angleYZ = yzAngleThreshold(angleYZ);
    }
  }

  const angleFactor = 0.3;
  const axisYOffset = 0.1;

  const pointerX = convertRange(1 - middleMcp.x + distToScreen * angleXZ * angleFactor, 0.1, 0.9, 0, 1);
  const pointerY = convertRange(middleMcp.y - distToScreen * angleYZ * angleFactor - axisYOffset, 0.2, 0.8, 0, 1);

  let result = {
    x: fx.filter(pointerX, timestamp) * window.screen.width,
    y: fy.filter(pointerY, timestamp) * window.screen.height,
    pinch: pinch
  }

  debugElement.innerText =
    `Screen Distance: ${distToScreen}
AngleXZ: ${Math.atan(angleXZ) * (180 / Math.PI)}
AngleYZ: ${Math.atan(angleYZ) * (180 / Math.PI)}
Pinched: ${result.pinch}, Distance: ${relativeDistance}
Pointer (x): ${thumbTip.x}
Pointer (y): ${thumbTip.y}
Pointer - Screen (x): ${Math.floor(result.x)}
Pointer - Screen (y): ${Math.floor(result.y)}`;

  if (cursorEnabled) {
    await invoke("mouse_action", { x: Math.floor(result.x), y: Math.floor(result.y), pinch: result.pinch });
  }
}

async function predict() {
  canvasElement.style.width = video.videoWidth;
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

    await driver(results.landmarks, startTimeMs);
  }
  canvasCtx.restore();

  if (webcamRunning === true) {
    window.requestAnimationFrame(predict);
  }
}
