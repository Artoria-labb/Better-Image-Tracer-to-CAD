// =============== PROGRESS LOG ===============
const logBox = document.getElementById("log");
function appendLog(msg) {
  const t = new Date().toLocaleTimeString();
  logBox.textContent += `[${t}] ${msg}\n`;
  logBox.scrollTop = logBox.scrollHeight; // auto-scroll
}

// =============== ELEMENTS ===================
const fileInput = document.getElementById("fileInput");
const traceBtn = document.getElementById("traceBtn");
const exportDxfBtn = document.getElementById("exportDxfBtn");
const downloadPreviewPng = document.getElementById("downloadPreviewPng");
const scaleInput = document.getElementById("scaleInput");

const fullCanvas = document.getElementById("fullCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const fullCtx = fullCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

let tracedPolylines = [];

// ================= HELPER ===================
function setCanvasSize(canvas, w, h) {
  canvas.width = w;
  canvas.height = h;
  const maxW = Math.min(window.innerWidth * 0.45, 1200);
  canvas.style.width = Math.min(maxW, w) + "px";
  canvas.style.height = (h * (parseFloat(canvas.style.width) / w)) + "px";
}

// =============== LOAD FILE ==================
async function renderFile(file) {
  appendLog("Loading file...");

  if (file.name.toLowerCase().endsWith(".pdf")) {
    appendLog("Rendering PDF…");

    const buffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
    const page = await pdf.getPage(1);

    const viewport = page.getViewport({ scale: 2 });
    setCanvasSize(fullCanvas, viewport.width, viewport.height);

    await page.render({ canvasContext: fullCtx, viewport }).promise;
    appendLog("PDF rendered to canvas.");
  } else {
    appendLog("Rendering image…");

    await new Promise((resolve) => {
      const r = new FileReader();
      const img = new Image();
      r.onload = (e) => { img.src = e.target.result; };
      img.onload = () => {
        setCanvasSize(fullCanvas, img.width, img.height);
        fullCtx.drawImage(img, 0, 0);
        appendLog("Image drawn to canvas.");
        resolve();
      };
      r.readAsDataURL(file);
    });
  }
}

// =============== TRACING ====================
function traceAll() {
  if (!fullCanvas.width) {
    appendLog("❌ No GA loaded.");
    return;
  }
  if (!window.cv) {
    appendLog("❌ OpenCV not ready yet.");
    return;
  }

  appendLog("Starting trace…");
  tracedPolylines = [];

  overlayCanvas.width = fullCanvas.width;
  overlayCanvas.height = fullCanvas.height;
  overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);

  const img = fullCtx.getImageData(0,0,fullCanvas.width,fullCanvas.height);
  let src = cv.matFromImageData(img);

  appendLog("Converting to grayscale…");
  let gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  appendLog("Thresholding...");
  let bw = new cv.Mat();
  cv.adaptiveThreshold(gray, bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 7);

  appendLog("Cleaning noise…");
  let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3,3));
  let closed = new cv.Mat();
  cv.morphologyEx(bw, closed, cv.MORPH_CLOSE, kernel);

  appendLog("Edge detection…");
  let edges = new cv.Mat();
  cv.Canny(closed, edges, 50, 200);

  appendLog("Finding contours…");
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);

  appendLog(`Contours found: ${contours.size()}`);

  appendLog("Simplifying + storing polylines…");

  for (let i = 0; i < contours.size(); i++) {
    let cnt = contours.get(i);

    // skip tiny ones
    if (cnt.data32S.length < 6) continue;

    let poly = [];
    for (let j = 0; j < cnt.data32S.length; j += 2) {
      poly.push({ x: cnt.data32S[j], y: cnt.data32S[j+1] });
    }
    if (poly.length >= 2) tracedPolylines.push(poly);
  }

  appendLog(`Final polyline count: ${tracedPolylines.length}`);

  // draw preview
  overlayCtx.strokeStyle = "#00FF88";
  overlayCtx.lineWidth = 1.2;
  tracedPolylines.forEach(poly => {
    overlayCtx.beginPath();
    overlayCtx.moveTo(poly[0].x, poly[0].y);
    for (let i=1;i<poly.length;i++) overlayCtx.lineTo(poly[i].x, poly[i].y);
    overlayCtx.stroke();
  });

  appendLog("Trace complete.");
}

// =============== DXF EXPORT ==================
function exportDXF() {
  if (tracedPolylines.length === 0) {
    appendLog("❌ Nothing to export.");
    return;
  }

  appendLog("Exporting DXF…");

  let dxf = "0\nSECTION\n2\nENTITIES\n";
  const scale = parseFloat(scaleInput.value) || 1.0;

  tracedPolylines.forEach(poly => {
    dxf += "0\nLWPOLYLINE\n90\n" + poly.length + "\n70\n0\n";
    poly.forEach(pt => {
      dxf += "10\n" + (pt.x * scale) + "\n20\n" + (pt.y * scale) + "\n";
    });
  });

  dxf += "0\nENDSEC\n0\nEOF";

  const blob = new Blob([dxf], { type:"application/dxf" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "traced.dxf";
  a.click();

  appendLog("DXF exported.");
}

// =============== EVENTS ======================
fileInput.addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  await renderFile(file);
  appendLog("Ready to trace.");
});

traceBtn.addEventListener("click", traceAll);
exportDxfBtn.addEventListener("click", exportDXF);

downloadPreviewPng.addEventListener("click", () => {
  appendLog("Downloading overlay preview…");

  const t = document.createElement("canvas");
  t.width = overlayCanvas.width;
  t.height = overlayCanvas.height;
  const tctx = t.getContext("2d");

  tctx.drawImage(fullCanvas,0,0);
  tctx.drawImage(overlayCanvas,0,0);

  const url = t.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = "preview.png";
  a.click();

  appendLog("Preview downloaded.");
});

