/* Browser-only capture; finished recordings use the existing video analysis endpoint. */
function setupWebcam(onSelect, onBusy) {
  const preview = document.querySelector("#webcam-preview");
  const open = document.querySelector("#webcam-open");
  const record = document.querySelector("#webcam-record");
  const stop = document.querySelector("#webcam-stop");
  const cancel = document.querySelector("#webcam-cancel");
  const status = document.querySelector("#webcam-status");
  const error = document.querySelector("#webcam-error");
  const empty = document.querySelector("#cam-empty");
  const rec = document.querySelector("#webcam-rec");
  const recTime = document.querySelector("#webcam-rec-time");
  const recBar = document.querySelector("#webcam-rec-bar");
  const hint = "Record 3–10 seconds with one clearly visible face. No audio is recorded.";
  const maxSeconds = 30;
  let stream = null, recorder = null, timer = null, playbackURL = null, session = 0;
  const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  function showPreview(on) {
    preview.hidden = !on;
    if (empty) empty.hidden = on;
  }

  function showRec(elapsed) {
    const on = elapsed !== null;
    if (rec) rec.hidden = !on;
    if (recBar) recBar.hidden = !on;
    if (!on) return;
    if (recTime) recTime.textContent = `${clock(elapsed)} / ${clock(maxSeconds)}`;
    if (recBar) recBar.firstElementChild.style.width = Math.min(100, elapsed / maxSeconds * 100) + "%";
  }

  function releaseCamera() {
    clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    preview.srcObject = null;
  }

  function reset() {
    session++; // Ignore a late permission response or a cancelled recorder's events.
    if (recorder && recorder.state !== "inactive") recorder.stop();
    recorder = null;
    releaseCamera();
    preview.pause();
    if (playbackURL) URL.revokeObjectURL(playbackURL);
    playbackURL = null;
    preview.removeAttribute("src");
    preview.controls = false;
    showPreview(false);
    showRec(null);
    open.hidden = false;
    open.textContent = "Enable webcam";
    record.hidden = stop.hidden = cancel.hidden = true;
    error.hidden = true;
    status.textContent = hint;
    onBusy(false);
  }

  function fail(message) {
    reset();
    error.textContent = message;
    error.hidden = false;
  }

  open.onclick = async () => {
    reset();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      fail("Webcam access requires localhost or HTTPS. Open this demo at http://127.0.0.1:7860, or upload a video.");
      return;
    }
    if (!window.MediaRecorder) {
      fail("This browser cannot record video. Try a current Chrome, Edge, Firefox or Safari, or upload a video.");
      return;
    }
    const current = session;
    open.hidden = true;
    cancel.hidden = false;
    onBusy(true);
    status.textContent = "Waiting for camera permission…";
    try {
      const acquired = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: "user", width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 25}},
        audio: false
      });
      if (current !== session) {
        acquired.getTracks().forEach(track => track.stop());
        return;
      }
      stream = acquired;
      preview.srcObject = stream;
      showPreview(true);
      await preview.play();
      if (current !== session) return;
      record.hidden = false;
      status.textContent = "Camera ready. Face the camera, then start recording (maximum 30 seconds).";
      stream.getVideoTracks().forEach(track => track.addEventListener("ended", () => {
        if (current !== session) return;
        if (recorder?.state === "recording") stop.click();
        else if (!recorder) fail("The camera disconnected. Reconnect it and try again.");
      }));
    } catch (err) {
      if (current !== session) return;
      const messages = {
        NotAllowedError: "Camera access was denied. Allow camera access in your browser and try again, or upload a video.",
        NotFoundError: "No webcam was found. Connect a camera or upload a video.",
        NotReadableError: "The webcam could not be opened. Close other apps using it and try again."
      };
      fail(messages[err.name] || "The webcam could not be started. Check your camera and browser permissions, then try again.");
    }
  };

  record.onclick = () => {
    if (!stream || recorder) return;
    const current = session;
    const chunks = [];
    const mimeType = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm", "video/mp4"]
      .find(type => MediaRecorder.isTypeSupported(type));
    try {
      const capture = new MediaRecorder(stream, mimeType ? {mimeType} : {});
      recorder = capture;
      capture.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      capture.onerror = () => {
        if (current === session) fail("Video recording failed. Try recording again or upload a video.");
      };
      capture.onstop = () => {
        if (current !== session) return;
        releaseCamera();
        recorder = null;
        const type = capture.mimeType || chunks[0]?.type || mimeType || "video/webm";
        const blob = new Blob(chunks, {type});
        if (!blob.size) {
          fail("The recording is empty. Try recording for at least 3 seconds.");
          return;
        }
        const extension = type.includes("mp4") ? "mp4" : "webm";
        const file = new File([blob], `webcam-${Date.now()}.${extension}`, {type});
        playbackURL = URL.createObjectURL(blob);
        preview.src = playbackURL;
        preview.controls = true;
        preview.autoplay = false;
        showPreview(true);
        showRec(null);
        open.hidden = false;
        open.textContent = "Record again";
        stop.hidden = cancel.hidden = true;
        onBusy(false);
        onSelect(file);
        status.textContent = "Recording ready. Preview it, then choose Analyze video, or record again.";
      };
      capture.start();
      record.hidden = true;
      stop.hidden = false;
      stop.disabled = false;
      const started = performance.now();
      const tick = () => {
        const elapsed = (performance.now() - started) / 1000;
        status.textContent = "Recording · keep your face clearly visible.";
        showRec(elapsed);
        if (elapsed >= maxSeconds) stop.click();
      };
      tick();
      timer = setInterval(tick, 200);
    } catch (err) {
      fail("Video recording could not be started. Try another browser or upload a video.");
    }
  };

  stop.onclick = () => {
    if (recorder?.state !== "recording") return;
    clearInterval(timer);
    stop.disabled = true;
    showRec(null);
    status.textContent = "Preparing recording…";
    recorder.stop();
    releaseCamera();
  };
  cancel.onclick = reset;
  window.addEventListener("pagehide", reset);
  return {reset};
}
