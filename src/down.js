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

  video.style.transform = "scaleX(-1) scaleY(-1)";
  canvasElement.style.transform = "scaleX(-1) scaleY(-1)";

  toggleCursorElement.addEventListener('click', () => {
    cursorEnabled = !cursorEnabled;
    if (cursorEnabled) {
      toggleCursorElement.innerText = 'Disable Cursor'
    } else {
      toggleCursorElement.innerText = 'Enable Cursor'
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

const PINCH_DISTANCE_THRESHOLD = 9;
const PINCH_BUFFER_SIZE = 9;
const PINCH_BUFFER_TOLERANCE = 0.5

// freq: camera fps
// mincutoff (start with 1): lower to reduce jitter
// beta (start with 0): increase beta to minimize lag
// dcutoff: keep as default

const fx = new OneEuroFilter(60, 0.001, 0.7, 1);
const fy = new OneEuroFilter(60, 0.001, 0.7, 1);

async function driver(landmarks, timestamp) {
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
    x: fx.filter(pointerX, timestamp) * window.screen.width,
    y: fy.filter(pointerY, timestamp) * window.screen.height,
    pinch: relativeDistance <= PINCH_DISTANCE_THRESHOLD
  }

  if (buffer.length < PINCH_BUFFER_SIZE) {
    buffer.push(result.pinch)
  } else {
    buffer.shift();
    buffer.push(result.pinch)

    result.pinch = buffer.reduce((a, b) => a + b, 0) >= (PINCH_BUFFER_TOLERANCE * PINCH_BUFFER_SIZE)
  }

  debugElement.innerText = 
  `Pinch Distance: ${relativeDistance}
Pinched: ${result.pinch}
Pointer (x): ${thumbTip.x}
Pointer (y): ${thumbTip.y}
Pointer - Screen (x): ${Math.floor(result.x)}
Pointer - Screen (y): ${Math.floor(result.y)}`;

  await invoke("mouse_action", { x: Math.floor(result.x), y: Math.floor(result.y), pinch: result.pinch });
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

    if (cursorEnabled) {
      await driver(results.landmarks, startTimeMs);
    }
  }
  canvasCtx.restore();

  if (webcamRunning === true) {
    window.requestAnimationFrame(predict);
  }
}
