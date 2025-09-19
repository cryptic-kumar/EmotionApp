// monitor.js — handles real-time camera -> /predict, overlay, multi-face, recommendations, history chart

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const frameCanvas = document.getElementById("frame");
const ctxFrame = frameCanvas.getContext("2d");
const octx = overlay.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const intervalInput = document.getElementById("interval");
const statusEl = document.getElementById("status");
const predictionsEl = document.getElementById("predictions");
const recommendationEl = document.getElementById("recommendation");

let stream = null;
let timer = null;
let running = false;
let history = [];

// Chart setup
const historyCtx = document.getElementById("historyChart").getContext("2d");
const chart = new Chart(historyCtx, {
  type: "bar",
  data: {
    labels: [],
    datasets: [{ label: "detections", data: [], backgroundColor: [] }],
  },
  options: {
    plugins: {
      tooltip: {
        callbacks: {
          label: (ctx) => `Emotion: ${ctx.raw.emotion}`,
        },
      },
      legend: { display: false },
    },
    scales: { y: { display: false } },
  },
});

// recommendations map
const RECOMMENDATIONS = {
  Happy: "You're looking great — play an upbeat song!",
  Sad: "Try a short walk, or watch something uplifting.",
  Angry: "Pause and take 3 slow breaths.",
  Neutral: "You're calm — try a focused work session.",
  Surprise: "Check a fun fact or a quick puzzle.",
  Fear: "Grounding: name 5 things you see, 4 you can touch.",
  Disgust: "Play a relaxing or funny clip to shift mood.",
};

function setOverlaySize() {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
}

async function startCamera() {
  try {
    if (window.isSecureContext === false && location.hostname !== "localhost") {
      // mobile browsers may require HTTPS
      alert(
        "Camera requires HTTPS on this device. Use localhost or an HTTPS tunnel (ngrok)."
      );
    }
    const md = navigator.mediaDevices;
    if (!md || !md.getUserMedia) throw new Error("getUserMedia not supported.");
    stream = await md.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = stream;
    await video.play();
    setOverlaySize();
    statusEl.innerText = "Camera started";
  } catch (e) {
    alert("Camera error: " + e.message);
    statusEl.innerText = "Camera error";
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  stream = null;
  video.pause();
  video.srcObject = null;
  clearInterval(timer);
  timer = null;
  running = false;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  statusEl.innerText = "Stopped";
}

function drawOverlay(preds) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  preds.forEach((p) => {
    const [x, y, w, h] = p.box;
    octx.strokeStyle = "#ff66c4";
    octx.lineWidth = 2;
    octx.strokeRect(x, y, w, h);
    octx.fillStyle = "#ff66c4";
    octx.font = "16px sans-serif";
    octx.fillText(
      `${p.label} ${(p.confidence * 100).toFixed(1)}%`,
      x + 6,
      Math.max(18, y - 6)
    );
  });
}

function updateUI(preds) {
  if (!preds || preds.length === 0) {
    predictionsEl.innerHTML = "No face detected";
    recommendationEl.innerHTML =
      "Point your camera to a face; ensure good lighting.";
    octx.clearRect(0, 0, overlay.width, overlay.height);
    return;
  }
  drawOverlay(preds);
  predictionsEl.innerHTML = preds
    .map(
      (p, i) =>
        `${i + 1}) <b>${p.label}</b> ${(p.confidence * 100).toFixed(1)}%`
    )
    .join("<br/>");
  recommendationEl.innerHTML = preds
    .map(
      (p) =>
        `<div class="bubble"><b>${p.label}</b>: ${
          RECOMMENDATIONS[p.label] || ""
        }</div>`
    )
    .join("");

  // history: push top face label
  const top = preds[0].label;
  const t = new Date().toLocaleTimeString();
  history.push({ time: t, label: top });
  if (history.length > 12) history.shift();

  chart.data.labels = history.map((h) => h.time);
  chart.data.datasets[0].data = history.map((h) => ({
    x: h.time,
    y: 1,
    emotion: h.label,
  }));
  chart.data.datasets[0].backgroundColor = history.map(() => "#ff66c4");
  chart.update();
}

async function captureAndSend() {
  if (!video || video.paused || video.readyState < 2) return;
  // draw frame to hidden canvas at video size
  ctxFrame.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  const dataURL = frameCanvas.toDataURL("image/jpeg", 0.8);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
    });
    if (!res.ok) {
      console.warn("Server returned", res.status);
      return;
    }
    const json = await res.json();
    if (json.predictions) updateUI(json.predictions);
  } catch (err) {
    console.error("Fetch error:", err);
  }
}

function startLoop() {
  if (running) return;
  const ms = Math.max(150, parseInt(intervalInput.value) || 700);
  timer = setInterval(captureAndSend, ms);
  running = true;
  statusEl.innerText = "Running";
}

startBtn.onclick = async () => {
  await startCamera();
  startLoop();
};

stopBtn.onclick = () => {
  stopCamera();
};
