let stream = null;
let timer = null;
let running = false;

const video = () => document.getElementById("video");
const canvas = () => document.getElementById("canvas");
const button = () => document.getElementById("toggle");
const status = () => document.getElementById("status");

function sendFrame() {
  if (!running || !video().videoWidth) return;
  const target = canvas();
  target.width = video().videoWidth;
  target.height = video().videoHeight;
  target.getContext("2d").drawImage(video(), 0, 0);
  Streamlit.setComponentValue(target.toDataURL("image/jpeg", 0.85));
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video().srcObject = stream;
    await video().play();
    running = true;
    button().textContent = "Pause live detection";
    status().textContent = "Camera is running.";
    timer = window.setInterval(sendFrame, 500);
  } catch (error) {
    status().textContent = "Camera permission or device error: " + error.message;
  }
}

function stop() {
  running = false;
  if (timer) window.clearInterval(timer);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video().srcObject = null;
  button().textContent = "Start live detection";
  status().textContent = "Camera is stopped.";
}

function render(event) {
  const args = event.detail.args;
  video().width = args.width;
  video().height = args.height;
  button().onclick = () => (running ? stop() : start());
  Streamlit.setFrameHeight(args.height + 80);
  if (!running && !stream) {
    start();
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, render);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(610);
