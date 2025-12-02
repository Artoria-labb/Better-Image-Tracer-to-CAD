// =========================
// Plan Tracer — Engineering Precision
// =========================

// ---------- Logging ----------
const logBox = document.getElementById("log");
function appendLog(msg) {
  const t = new Date().toLocaleTimeString();
  logBox.textContent += `[${t}] ${msg}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}

// ---------- Elements ----------
const fileInput = document.getElementById("fileInput");
const traceBtn = document.getElementById("traceBtn");
const exportDxfBtn = document.getElementById("exportDxfBtn");
const downloadPreviewPng = document.getElementById("downloadPreviewPng");
const scaleInput = document.getElementById("scaleInput");

const fullCanvas = document.getElementById("fullCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const fullCtx = fullCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

let detectedLines = []; // array of {x1,y1,x2,y2}
let imageLoaded = false;

// ---------- Utilities ----------
function setCanvasSize(canvas, w, h) {
  canvas.width = w;
  canvas.height = h;
  const maxW = Math.min(window.innerWidth * 0.45, 1200);
  canvas.style.width = Math.min(maxW, w) + "px";
  canvas.style.height = (h * (parseFloat(canvas.style.width) / w)) + "px";
}

// ---------- Render file (pdf/image) ----------
async function renderFile(file) {
  appendLog("Loading file...");
  imageLoaded = false;
  if (file.name.toLowerCase().endsWith(".pdf")) {
    appendLog("Rendering PDF (page 1) …");
    const buffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
    const page = await pdf.getPage(1);
    const viewport = page.getViewport({ scale: 2.0 });
    setCanvasSize(fullCanvas, Math.floor(viewport.width), Math.floor(viewport.height));
    await page.render({ canvasContext: fullCtx, viewport }).promise;
    appendLog("PDF rendered.");
  } else {
    appendLog("Rendering image …");
    await new Promise((resolve, reject) => {
      const reader = new FileReader();
      const img = new Image();
      reader.onload = (e) => img.src = e.target.result;
      img.onload = () => {
        setCanvasSize(fullCanvas, img.width, img.height);
        fullCtx.drawImage(img, 0, 0);
        appendLog("Image drawn to canvas.");
        resolve();
      };
      img.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
  // prepare overlay
  overlayCanvas.width = fullCanvas.width;
  overlayCanvas.height = fullCanvas.height;
  overlayCanvas.style.width = fullCanvas.style.width;
  overlayCanvas.style.height = fullCanvas.style.height;
  overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
  imageLoaded = true;
  appendLog("Ready for trace.");
}

// ---------- Zhang-Suen thinning implementation ----------
function zhangSuenThinning(binary, w, h) {
  // binary: Uint8Array (0 or 1) row-major
  // returns a new Uint8Array of same size thinned
  const idx = (x,y) => y*w + x;
  let img = new Uint8Array(binary); // copy
  let changing = true;

  function neighbors(x,y) {
    // 8 neighbors in order N, NE, E, SE, S, SW, W, NW
    return [
      img[idx(x, y-1)] || 0,
      img[idx(x+1, y-1)] || 0,
      img[idx(x+1, y)] || 0,
      img[idx(x+1, y+1)] || 0,
      img[idx(x, y+1)] || 0,
      img[idx(x-1, y+1)] || 0,
      img[idx(x-1, y)] || 0,
      img[idx(x-1, y-1)] || 0
    ];
  }
  function A(nei) {
    // number of 0->1 transitions in sequence
    let cnt = 0;
    for (let i=0;i<nei.length;i++){
      const cur = nei[i];
      const nxt = nei[(i+1)%nei.length];
      if (cur === 0 && nxt === 1) cnt++;
    }
    return cnt;
  }
  function B(nei) { // number of 1 neighbors
    return nei.reduce((s,v)=>s+v,0);
  }

  while (changing) {
    changing = false;
    let toRemove = [];
    // Step 1
    for (let y=1;y<h-1;y++){
      for (let x=1;x<w-1;x++){
        const p = img[idx(x,y)];
        if (p !== 1) continue;
        const nei = neighbors(x,y);
        const b = B(nei);
        const a = A(nei);
        if (b >= 2 && b <= 6 && a === 1 &&
            (nei[0] * nei[2] * nei[4] === 0) &&
            (nei[2] * nei[4] * nei[6] === 0)) {
          toRemove.push(idx(x,y));
        }
      }
    }
    if (toRemove.length) {
      changing = true;
      for (let k of toRemove) img[k] = 0;
      toRemove = [];
    }

    // Step 2
    for (let y=1;y<h-1;y++){
      for (let x=1;x<w-1;x++){
        const p = img[idx(x,y)];
        if (p !== 1) continue;
        const nei = neighbors(x,y);
        const b = B(nei);
        const a = A(nei);
        if (b >= 2 && b <= 6 && a === 1 &&
            (nei[0] * nei[2] * nei[6] === 0) &&
            (nei[0] * nei[4] * nei[6] === 0)) {
          toRemove.push(idx(x,y));
        }
      }
    }
    if (toRemove.length) {
      changing = true;
      for (let k of toRemove) img[k] = 0;
    }
  }

  return img;
}

// ---------- Convert ImageData -> binary (0/1) using adaptive threshold approx ----------
function imageDataToBinary(imgData, w, h) {
  const data = imgData.data;
  const out = new Uint8Array(w*h);
  // simple grayscale then adaptive-ish threshold by local mean (fast approximate)
  // We'll do block mean thresholding with block 15
  const block = 15;
  const half = Math.floor(block/2);
  // build integral image for fast mean
  const integral = new Float64Array((w+1)*(h+1));
  for (let y=0;y<h;y++){
    let rowSum = 0;
    for (let x=0;x<w;x++){
      const i = (y*w + x)*4;
      // grayscale luminosity
      const g = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
      rowSum += g;
      integral[(y+1)*(w+1) + (x+1)] = integral[(y)*(w+1)+(x+1)] + rowSum;
    }
  }
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      const x1 = Math.max(0, x-half);
      const x2 = Math.min(w-1, x+half);
      const y1 = Math.max(0, y-half);
      const y2 = Math.min(h-1, y+half);
      const area = (x2 - x1 + 1) * (y2 - y1 + 1);
      const sum = integral[(y2+1)*(w+1)+(x2+1)] - integral[(y1)*(w+1)+(x2+1)]
                - integral[(y2+1)*(w+1)+(x1)] + integral[(y1)*(w+1)+(x1)];
      const idx = (y*w + x)*4;
      const g = 0.299*data[idx] + 0.587*data[idx+1] + 0.114*data[idx+2];
      // threshold: pixel darker than local mean * (1 - k)
      const mean = sum / area;
      out[y*w + x] = (g < (mean * 0.95)) ? 1 : 0;
    }
  }
  return out;
}

// ---------- Merge collinear/close segments ----------
function mergeLines(lines, angleTolDeg=3, gapTol=6) {
  // lines: [{x1,y1,x2,y2},...]
  if (!lines || lines.length === 0) return [];
  // convert to normalized representation (center, angle, length)
  function angleDeg(a) { return a * 180/Math.PI; }
  function lineAngle(l) {
    return Math.atan2(l.y2 - l.y1, l.x2 - l.x1);
  }
  function dist(a,b){
    const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy);
  }
  let used = new Array(lines.length).fill(false);
  let groups = [];
  for (let i=0;i<lines.length;i++){
    if (used[i]) continue;
    used[i]=true;
    let group = [lines[i]];
    let merged = true;
    while (merged) {
      merged = false;
      for (let j=0;j<lines.length;j++){
        if (used[j]) continue;
        // compare with group's representative angle
        const rep = group[0];
        const a1 = lineAngle(rep), a2 = lineAngle(lines[j]);
        let diff = Math.abs(angleDeg(a1 - a2));
        diff = Math.min(diff, Math.abs(360 - diff));
        if (diff <= angleTolDeg) {
          // check end-to-end distance: if any endpoints are within gapTol
          const endsA = [
            {x:group[0].x1,y:group[0].y1},
            {x:group[0].x2,y:group[0].y2}
          ];
          const endsB = [
            {x:lines[j].x1,y:lines[j].y1},
            {x:lines[j].x2,y:lines[j].y2}
          ];
          let close = false;
          for (let ea of endsA) for (let eb of endsB) {
            if (dist(ea, eb) <= gapTol) close = true;
          }
          if (close) {
            group.push(lines[j]);
            used[j] = true;
            merged = true;
          }
        }
      }
    }
    groups.push(group);
  }
  // now for each group, compute bounding line by projecting endpoints onto the group's main direction
  const mergedLines = groups.map(group => {
    // find best-fit line via simple averaging of angles
    let sx=0, sy=0;
    group.forEach(l => {
      sx += (l.x1 + l.x2)/2;
      sy += (l.y1 + l.y2)/2;
    });
    const mx = sx / group.length, my = sy / group.length;
    // compute average angle via vector sum
    let vx=0, vy=0;
    group.forEach(l => {
      const a = lineAngle(l);
      vx += Math.cos(a);
      vy += Math.sin(a);
    });
    const angle = Math.atan2(vy, vx);
    // project all endpoints onto this direction and take min/max projection
    const dirx = Math.cos(angle), diry = Math.sin(angle);
    let projections = [];
    group.forEach(l => {
      [ {x:l.x1,y:l.y1}, {x:l.x2,y:l.y2} ].forEach(p => {
        const proj = p.x*dirx + p.y*diry;
        projections.push({proj,p});
      });
    });
    projections.sort((a,b)=>a.proj-b.proj);
    const p0 = projections[0].p;
    const p1 = projections[projections.length-1].p;
    return { x1: Math.round(p0.x), y1: Math.round(p0.y), x2: Math.round(p1.x), y2: Math.round(p1.y) };
  });
  return mergedLines;
}

// ---------- Main trace function (centerline -> Hough -> merge) ----------
async function traceCenterlines() {
  if (!imageLoaded) { appendLog("No image loaded."); return; }
  if (!window.OPENCV_READY || typeof cv === "undefined") { appendLog("OpenCV not ready."); return; }

  appendLog("Starting engineering-precision trace...");
  appendLog("Getting image data...");
  const w = fullCanvas.width, h = fullCanvas.height;
  const imgData = fullCtx.getImageData(0,0,w,h);

  appendLog("Binarizing (local mean)...");
  const bin = imageDataToBinary(imgData, w, h); // Uint8Array (0/1)

  appendLog("Applying Zhang-Suen thinning (skeletonization) — this may take a few seconds...");
  const thin = zhangSuenThinning(bin, w, h); // Uint8Array centerlines

  appendLog("Creating OpenCV Mat from thinned image...");
  // Convert to Uint8Array 0/255
  const arr = new Uint8Array(w*h);
  for (let i=0;i<w*h;i++) arr[i] = thin[i] ? 255 : 0;

  // create cv.Mat from raw data
  let mat = cv.matFromArray(h, w, cv.CV_8UC1, arr);
  appendLog("Morphological clean (optional)...");
  let cleaned = new cv.Mat();
  let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1,1));
  cv.morphologyEx(mat, cleaned, cv.MORPH_OPEN, kernel);

  appendLog("Running HoughLinesP to detect straight segments...");
  // Parameters tuned for engineering precision; you can tweak: rho, theta, threshold, minLineLen, maxGap
  const rho = 1;
  const theta = Math.PI / 180;
  const threshold = Math.max(30, Math.round(Math.min(w,h) / 60)); // stronger requirement for longer/clear segments
  const minLineLen = Math.max(20, Math.round(Math.min(w,h) / 50));
  const maxGap = Math.max(6, Math.round(Math.min(w,h) / 500));
  appendLog(`Hough params: threshold=${threshold}, minLen=${minLineLen}, maxGap=${maxGap}`);
  let lines = new cv.Mat();
  cv.HoughLinesP(cleaned, lines, rho, theta, threshold, minLineLen, maxGap);

  appendLog(`Raw Hough segments found: ${lines.rows}`);

  // convert to JS array
  let rawLines = [];
  for (let i=0;i<lines.rows;i++){
    const x1 = lines.data32S[i*4], y1 = lines.data32S[i*4+1],
          x2 = lines.data32S[i*4+2], y2 = lines.data32S[i*4+3];
    rawLines.push({x1,y1,x2,y2});
  }

  appendLog("Merging collinear/near segments...");
  const merged = mergeLines(rawLines, 2.5, Math.max(6, Math.round(Math.min(w,h)/400)));

  appendLog(`Merged lines: ${merged.length}`);

  // store for export and preview
  detectedLines = merged;

  // draw preview overlay
  overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
  // draw skeleton faint
  overlayCtx.strokeStyle = "rgba(255,90,90,0.12)";
  overlayCtx.lineWidth = 1;
  // optional: draw some skeleton pixels (skip for performance)

  // draw final merged lines
  overlayCtx.strokeStyle = "#00ff88";
  overlayCtx.lineWidth = Math.max(1, Math.round(w / 1200));
  for (let L of detectedLines) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(L.x1 + 0.5, L.y1 + 0.5);
    overlayCtx.lineTo(L.x2 + 0.5, L.y2 + 0.5);
    overlayCtx.stroke();
  }

  // cleanup mats
  mat.delete(); cleaned.delete(); kernel.delete(); lines.delete();

  appendLog("Centerline trace complete. Clean CAD lines available for export.");
}

// ---------- DXF export (lines only, no image) ----------
function buildDXFFromLines(lines) {
  // Build header compatible with AutoCAD (AC1027 -> 2013)
  let dxf = "";
  dxf += "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n0\nENDSEC\n";
  dxf += "0\nSECTION\n2\nTABLES\n0\nENDSEC\n";
  dxf += "0\nSECTION\n2\nENTITIES\n";
  const scale = parseFloat(scaleInput.value) || 1.0;

  for (let i=0;i<lines.length;i++){
    const L = lines[i];
    // LINE entity is simplest (two vertex)
    const x1 = (L.x1 * scale).toFixed(6);
    const y1 = ((fullCanvas.height - L.y1) * scale).toFixed(6); // flip y for CAD
    const x2 = (L.x2 * scale).toFixed(6);
    const y2 = ((fullCanvas.height - L.y2) * scale).toFixed(6);
    dxf += "0\nLINE\n8\nTRACED\n10\n" + x1 + "\n20\n" + y1 + "\n30\n0.0\n11\n" + x2 + "\n21\n" + y2 + "\n31\n0.0\n";
  }

  dxf += "0\nENDSEC\n0\nEOF";
  return dxf;
}

function downloadText(filename, text, mime="application/dxf") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(()=>URL.revokeObjectURL(url), 1000);
}

// ---------- Events ----------
fileInput.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  try {
    await renderFile(f);
  } catch (err) {
    appendLog("Error rendering file: " + (err.message || err));
  }
});

traceBtn.addEventListener("click", async () => {
  try {
    appendLog("Trace button pressed.");
    await traceCenterlines();
  } catch (err) {
    appendLog("Error during trace: " + (err.message || err));
  }
});

exportDxfBtn.addEventListener("click", () => {
  if (!detectedLines || detectedLines.length === 0) {
    appendLog("No lines detected. Run trace first.");
    return;
  }
  appendLog("Preparing DXF...");
  const dxf = buildDXFFromLines(detectedLines);
  downloadText("traced_clean.dxf", dxf);
  appendLog("DXF ready — download started.");
});

downloadPreviewPng.addEventListener("click", () => {
  appendLog("Preparing preview PNG...");
  const tmp = document.createElement("canvas");
  tmp.width = fullCanvas.width;
  tmp.height = fullCanvas.height;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(fullCanvas, 0, 0);
  tctx.drawImage(overlayCanvas, 0, 0);
  const url = tmp.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = "preview_traced.png";
  a.click();
  appendLog("Preview PNG downloaded.");
});

// Optional: expose OpenCV ready
window.onOpenCvReady = function() {
  appendLog("OpenCV ready.");
  // sometimes async: ensure cv exists
  if (!window.cv) appendLog("Warning: cv not found after onOpenCvReady.");
};
