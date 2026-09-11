function sendMessageToStreamlitClient(type, data) {
  const outData = Object.assign({isStreamlitMessage: true, type: type}, data);
  window.parent.postMessage(outData, "*");
}
const Streamlit = {
  setComponentReady: () => sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1}),
  setFrameHeight: (height) => sendMessageToStreamlitClient("streamlit:setFrameHeight", {height}),
  setComponentValue: (value) => sendMessageToStreamlitClient("streamlit:setComponentValue", {value}),
  RENDER_EVENT: "streamlit:render",
  events: {addEventListener: (type, callback) => window.addEventListener("message", (event) => {
    if (event.data.type === type) {
      event.detail = event.data;
      callback(event);
    }
  })},
};
