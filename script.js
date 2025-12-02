// ===================== Plan Tracer: Lines + Arcs + Circles =====================
// Full file. Replace your existing script.js with this one.
// =================================================================================

// --------------------- Logging ---------------------
const logEl = document.getElementById("log");
function appendLog(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}
appendLog("App loaded.");

// --------------------- Elements ---------------------
const fileInput = document.getElementById("fileInput");
const detectBtn = document.getElementById("detectBtn");
const exportBtn = document.getElementById("exportBtn");
const previewBtn = document.getElementById("previewBtn");
const scaleInput = document.getElementById("scaleInput");

const fullCanvas = document.getElementById("fullCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const fullCtx = fullCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

let imgLoaded = false;
let detected = { lines: [], circles: [], arcs: [] };

// --------------------- Helpers ---------------------
function setCanvasSize(canvas, w, h) {
  canvas.width = w; canvas.height = h;
  const maxW = Math.min(window.innerWidth * 0.45, 1400);
  canvas.style.width = Math.min(maxW, w) + "px";
  canvas.style.height = (h * (parseFloat(canvas.style.width) / w)) + "px";
}

// safe numeric
function toNum(v, fallback = 1) {
  const n = parseFloat(v);
  return (isFinite(n) && n > 0) ? n : fallback;
}

// --------------------- File Rendering ---------------------
async function renderFile(file) {
  appendLog("Loading file: " + file.name);
  imgLoaded = false;

  if (file.name.toLowerCase().endsWith(".pdf")) {
    appendLog("Rendering PDF page 1...");
    const buf = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    const page = await pdf.getPage(1);
    const viewport = page.getViewport({ scale: 2.0 });
    setCanvasSize(fullCanvas, Math.floor(viewport.width), Math.floor(viewport.height));
    await page.render({ canvasContext: fullCtx, viewport }).promise;
    appendLog("PDF rendered to canvas: " + fullCanvas.width + "x" + fullCanvas.height);
  } else {
    appendLog("Rendering image...");
    await new Promise((resolve, reject) => {
      const reader = new FileReader();
      const img = new Image();
      reader.onload = (e) => img.src = e.target.result;
      img.onload = () => {
        setCanvasSize(fullCanvas, img.width, img.height);
        fullCtx.drawImage(img, 0, 0);
        appendLog("Image drawn to canvas: " + img.width + "x" + img.height);
        resolve();
      };
      img.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // prepare overlay same size
  overlayCanvas.width = fullCanvas.width; overlayCanvas.height = fullCanvas.height;
  overlayCanvas.style.width = fullCanvas.style.width; overlayCanvas.style.height = fullCanvas.style.height;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  imgLoaded = true;
  appendLog("Ready for detection.");
}

// --------------------- Multi-scale Canny (combine edges) ---------------------
function multiScaleEdges(srcMat) {
  // srcMat: cv.Mat grayscale CV_8UC1
  // returns edgeMat CV_8UC1
  appendLog("Running multi-scale edge detection...");
  const scales = [1, 2.0, 3.5]; // blur sigmas (approx)
  let combined = new cv.Mat.zeros(srcMat.rows, srcMat.cols, cv.CV_8UC1);

  for (let s = 0; s < scales.length; s++) {
    const k = Math.max(3, Math.round(scales[s] * 3) | 1);
    const blurred = new cv.Mat();
    cv.GaussianBlur(srcMat, blurred, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);

    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 40, 150); // these thresholds are conservative
    // dilate slightly to connect
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2,2));
    cv.dilate(edges, edges, kernel);

    // combine
    cv.bitwise_or(combined, edges, combined);

    blurred.delete(); edges.delete(); kernel.delete();
  }

  // thin a bit with morphological thinning-like operation: skeleton via distance transform + threshold
  // (OpenCV.js doesn't have built-in morphological skeleton, we'll keep as-is)
  appendLog("Edge combination complete.");
  return combined;
}

// --------------------- Hough lines detection ---------------------
function detectHoughLines(edgeMat, minLenPx, maxGapPx, threshold) {
  appendLog("Detecting Hough lines (minLen=" + minLenPx + ", maxGap=" + maxGapPx + ", threshold=" + threshold + ")...");
  const linesMat = new cv.Mat();
  cv.HoughLinesP(edgeMat, linesMat, 1, Math.PI/180, threshold, minLenPx, maxGapPx);

  const lines = [];
  for (let i = 0; i < linesMat.rows; i++) {
    const x1 = linesMat.data32S[i*4], y1 = linesMat.data32S[i*4+1];
    const x2 = linesMat.data32S[i*4+2], y2 = linesMat.data32S[i*4+3];
    lines.push({ x1, y1, x2, y2 });
  }
  linesMat.delete();
  appendLog("Hough lines detected: " + lines.length);
  return lines;
}

// --------------------- Hough circles detection ---------------------
function detectHoughCircles(grayMat) {
  appendLog("Detecting circles (HoughCircles)...");
  // HoughCircles uses CV_8UC1 and returns float32 circles
  // Use dp=1.5 ; adapt minRadius,maxRadius to image size
  const dp = 1.5;
  const minDist = Math.round(Math.min(grayMat.rows, grayMat.cols) / 30);
  const minRadius = 6;
  const maxRadius = Math.round(Math.min(grayMat.rows, grayMat.cols) / 10);
  const circles = new cv.Mat();
  try {
    cv.HoughCircles(grayMat, circles, cv.HOUGH_GRADIENT, dp, minDist, 100, 30, minRadius, maxRadius);
  } catch (e) {
    appendLog("HoughCircles exception: " + e.message);
  }
  const out = [];
  if (circles.cols > 0) {
    for (let i = 0; i < circles.cols; i++) {
      const x = circles.data32F[i*3], y = circles.data32F[i*3+1], r = circles.data32F[i*3+2];
      out.push({ x: Math.round(x), y: Math.round(y), r: Math.round(r) });
    }
  }
  circles.delete();
  appendLog("Circles detected: " + out.length);
  return out;
}

// --------------------- Contour arcs approx ---------------------
function detectContoursAsArcs(edgeMat, areaThresh = 20) {
  appendLog("Detecting contours for arc approximation...");
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edgeMat, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
  const arcs = [];

  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = Math.abs(cv.contourArea(cnt));
    if (area < areaThresh) { cnt.delete(); continue; }

    // bounding box; if contour is curved-ish, try fitEllipse and decide arc/circle
    if (cnt.rows >= 6) {
      try {
        const ellipse = cv.fitEllipse(cnt);
        // ellipse: {center:{x,y}, size:{width,height}, angle}
        // if ellipse size and contour length suit -> add as arc candidate
        const arcThreshold = 0.6; // accept only if ellipse major/minor ratio not too extreme
        const ratio = Math.min(ellipse.size.width, ellipse.size.height) / Math.max(ellipse.size.width, ellipse.size.height);
        if (ratio > 0.5) {
          arcs.push({
            type: "ellipse",
            cx: Math.round(ellipse.center.x),
            cy: Math.round(ellipse.center.y),
            rx: Math.round(ellipse.size.width / 2),
            ry: Math.round(ellipse.size.height / 2),
            angle: ellipse.angle
          });
        } else {
          // for very stretched shapes, approximate by polyline (we'll export as many short lines)
          const approx = new cv.Mat();
          cv.approxPolyDP(cnt, approx, 2.0, false); // epsilon = 2 px
          const pts = [];
          for (let j = 0; j < approx.data32S.length; j += 2) {
            pts.push({ x: approx.data32S[j], y: approx.data32S[j+1] });
          }
          if (pts.length >= 3) {
            arcs.push({ type: "poly", pts });
          }
          approx.delete();
        }
      } catch (e) {
        // fitEllipse can fail; fallback to poly
        const approx = new cv.Mat();
        cv.approxPolyDP(cnt, approx, 2.0, false);
        const pts = [];
        for (let j = 0; j < approx.data32S.length; j += 2) {
          pts.push({ x: approx.data32S[j], y: approx.data32S[j+1] });
        }
        if (pts.length >= 3) arcs.push({ type: "poly", pts });
        approx.delete();
      }
    } else {
      // small contour; ignore or approximate as polyline
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 1.0, false);
      const pts = [];
      for (let j = 0; j < approx.data32S.length; j += 2) {
        pts.push({ x: approx.data32S[j], y: approx.data32S[j+1] });
      }
      if (pts.length >= 3) arcs.push({ type: "poly", pts });
      approx.delete();
    }
    cnt.delete();
  }

  contours.delete(); hierarchy.delete();
  appendLog("Arc candidates (contours) found: " + arcs.length);
  return arcs;
}

// --------------------- Merge and extend lines ---------------------
function mergeCollinear(lines, angleTolDeg = 2.5, gapTolPx = 6) {
  // group collinear by angle, then merge if endpoints are close
  if (!lines || lines.length === 0) return [];

  function angle(l) { return Math.atan2(l.y2 - l.y1, l.x2 - l.x1); }
  function deg(rad) { return rad * 180 / Math.PI; }
  function dist(a,b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  const used = new Array(lines.length).fill(false);
  const groups = [];

  for (let i = 0; i < lines.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    let group = [lines[i]];
    let expanded = true;
    while (expanded) {
      expanded = false;
      for (let j = 0; j < lines.length; j++) {
        if (used[j]) continue;
        for (let g = 0; g < group.length; g++) {
          const a1 = deg(angle(group[g])), a2 = deg(angle(lines[j]));
          let d = Math.abs(a1 - a2); d = Math.min(d, 360 - d);
          if (d <= angleTolDeg) {
            // check endpoint distances
            const endsGroup = [{x:group[g].x1, y:group[g].y1}, {x:group[g].x2, y:group[g].y2}];
            const endsJ = [{x:lines[j].x1, y:lines[j].y1}, {x:lines[j].x2, y:lines[j].y2}];
            let close = false;
            for (const ea of endsGroup) for (const eb of endsJ) if (dist(ea, eb) <= gapTolPx) close = true;
            if (close) { group.push(lines[j]); used[j] = true; expanded = true; break; }
          }
        }
      }
    }
    groups.push(group);
  }

  // collapse each group into single extended line by projecting endpoints onto main direction
  const merged = groups.map(group => {
    // compute average angle direction
    let sx = 0, sy = 0;
    group.forEach(l => {
      const a = angle(l);
      sx += Math.cos(a); sy += Math.sin(a);
    });
    const ang = Math.atan2(sy, sx);
    // direction vector
    const dx = Math.cos(ang), dy = Math.sin(ang);
    // project endpoints
    const projs = [];
    group.forEach(l => {
      projs.push({p:{x:l.x1,y:l.y1}, v: l.x1*dx + l.y1*dy});
      projs.push({p:{x:l.x2,y:l.y2}, v: l.x2*dx + l.y2*dy});
    });
    projs.sort((a,b)=>a.v-b.v);
    const p0 = projs[0].p, p1 = projs[projs.length-1].p;
    return { x1: Math.round(p0.x), y1: Math.round(p0.y), x2: Math.round(p1.x), y2: Math.round(p1.y) };
  });

  appendLog("Merged lines into: " + merged.length);
  return merged;
}

// --------------------- Main detection pipeline ---------------------
async function detectAll() {
  if (!imgLoaded) { appendLog("No image loaded."); return; }
  if (!window.OPENCV_READY || typeof cv === "undefined") { appendLog("OpenCV not ready."); return; }

  appendLog("Starting detection pipeline...");

  const w = fullCanvas.width, h = fullCanvas.height;
  appendLog("Canvas: " + w + "x" + h);

  // 1) get grayscale mat
  const imgData = fullCtx.getImageData(0,0,w,h);
  let srcMat = cv.matFromImageData(imgData);
  let gray = new cv.Mat();
  cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
  srcMat.delete();

  // 2) enhance contrast / denoise
  appendLog("Applying bilateral filter + CLAHE (local contrast) for clarity...");
  let den = new cv.Mat();
  cv.bilateralFilter(gray, den, 9, 75, 75);
  try {
    const clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
    let clahed = new cv.Mat();
    clahe.apply(den, clahed);
    den.delete();
    den = clahed;
    clahe.delete();
  } catch (e) {
    // CLAHE might not be present in all builds; skip if fails
    appendLog("CLAHE not available: " + (e.message || e));
  }

  // 3) multi-scale edges
  const edges = multiScaleEdges(den);

  // 4) detect circles
  const circles = detectHoughCircles(den);

  // 5) run HoughLinesP for lines (multi-scale thresholds)
  const minDim = Math.min(w,h);
  const linesAll = [];
  const scales = [
    {minLen: Math.max(20, Math.round(minDim/80)), maxGap: Math.max(6, Math.round(minDim/400)), thresh: Math.max(30, Math.round(minDim/200))},
    {minLen: Math.max(35, Math.round(minDim/60)), maxGap: Math.max(8, Math.round(minDim/300)), thresh: Math.max(45, Math.round(minDim/130))},
    {minLen: Math.max(60, Math.round(minDim/40)), maxGap: Math.max(10, Math.round(minDim/200)), thresh: Math.max(80, Math.round(minDim/100))}
  ];
  for (let s of scales) {
    const ls = detectHoughLines(edges, s.minLen, s.maxGap, s.thresh);
    linesAll.push(...ls);
  }

  appendLog("Total raw Hough segments (combined): " + linesAll.length);

  // 6) merge collinear segments
  const merged = mergeCollinear(linesAll, 2.0, Math.max(6, Math.round(minDim/500)));

  // 7) detect contours for arcs / complex curves (approx)
  // generate a clean binary image from edges for contour extraction
  appendLog("Preparing binary for contour arc detection...");
  const bin = new cv.Mat();
  cv.threshold(edges, bin, 1, 255, cv.THRESH_BINARY);
  const arcs = detectContoursAsArcs(bin, 18);
  bin.delete();

  // cleanup intermediate mats
  edges.delete(); den.delete(); gray.delete();

  // 8) assign detected objects
  detected.lines = merged;
  detected.circles = circles;
  detected.arcs = arcs;

  appendLog(`Detection complete. lines:${detected.lines.length} circles:${detected.circles.length} arcs:${detected.arcs.length}`);

  // 9) draw preview
  drawPreview();
}

// --------------------- Preview drawing ---------------------
function drawPreview() {
  overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);

  // draw lines
  overlayCtx.strokeStyle = "#00ff88";
  overlayCtx.lineWidth = Math.max(1, Math.round(fullCanvas.width / 1600));
  detected.lines.forEach(L => {
    overlayCtx.beginPath();
    overlayCtx.moveTo(L.x1 + 0.5, L.y1 + 0.5);
    overlayCtx.lineTo(L.x2 + 0.5, L.y2 + 0.5);
    overlayCtx.stroke();
  });

  // draw circles
  overlayCtx.strokeStyle = "#00ffff";
  detected.circles.forEach(c => {
    overlayCtx.beginPath();
    overlayCtx.arc(c.x, c.y, c.r, 0, Math.PI*2);
    overlayCtx.stroke();
  });

  // draw arc polylines / ellipse approx
  overlayCtx.strokeStyle = "#9effff";
  detected.arcs.forEach(a => {
    if (a.type === "poly") {
      overlayCtx.beginPath();
      overlayCtx.moveTo(a.pts[0].x, a.pts[0].y);
      for (let i=1;i<a.pts.length;i++) overlayCtx.lineTo(a.pts[i].x, a.pts[i].y);
      overlayCtx.stroke();
    } else if (a.type === "ellipse") {
      // approximate ellipse contour by drawing its bounding ellipse
      overlayCtx.save();
      overlayCtx.translate(a.cx, a.cy);
      overlayCtx.rotate((a.angle||0) * Math.PI/180);
      overlayCtx.beginPath();
      overlayCtx.scale(1, a.ry / (a.rx || a.ry));
      overlayCtx.arc(0,0, a.rx, 0, Math.PI*2);
      overlayCtx.restore();
      overlayCtx.stroke();
    }
  });

  appendLog("Preview drawn.");
}

// --------------------- DXF Builder (Rhino-compatible) ---------------------
function buildRhinoCompatibleDXF(lines, circles, arcs) {
  appendLog("Building Rhino-compatible DXF...");
  const scale = toNum(scaleInput.value, 1.0);

  // Basic header and layer table with 'TRACED' layer
  let dxf = "";
  dxf += "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n0\nENDSEC\n";
  dxf += "0\nSECTION\n2\nTABLES\n";
  dxf += "0\nTABLE\n2\nLAYER\n70\n1\n";
  dxf += "0\nLAYER\n2\nTRACED\n70\n0\n62\n7\n6\nCONTINUOUS\n";
  dxf += "0\nENDTAB\n";
  dxf += "0\nENDSEC\n";

  // BLOCKS (empty) - Rhino likes it present
  dxf += "0\nSECTION\n2\nBLOCKS\n0\nENDSEC\n";

  // ENTITIES
  dxf += "0\nSECTION\n2\nENTITIES\n";

  // Add LINE entities for straight lines
  for (let L of lines) {
    // flip Y to CAD bottom-left origin
    const x1 = (L.x1 * scale).toFixed(6);
    const y1 = ((fullCanvas.height - L.y1) * scale).toFixed(6);
    const x2 = (L.x2 * scale).toFixed(6);
    const y2 = ((fullCanvas.height - L.y2) * scale).toFixed(6);
    dxf += "0\nLINE\n8\nTRACED\n10\n" + x1 + "\n20\n" + y1 + "\n30\n0.0\n11\n" + x2 + "\n21\n" + y2 + "\n31\n0.0\n";
  }

  // Add CIRCLE entities for circles
  for (let C of circles) {
    const cx = (C.x * scale).toFixed(6);
    const cy = ((fullCanvas.height - C.y) * scale).toFixed(6);
    const r = (C.r * scale).toFixed(6);
    dxf += "0\nCIRCLE\n8\nTRACED\n10\n" + cx + "\n20\n" + cy + "\n30\n0.0\n40\n" + r + "\n";
  }

  // Add arcs/polylines from arcs array:
  // For ellipse type, approximate by many small LINE segments (good DXF compatibility)
  function addPoly(pts) {
    // export as LWPOLYLINE
    dxf += "0\nLWPOLYLINE\n8\nTRACED\n90\n" + pts.length + "\n70\n0\n";
    for (let p of pts) {
      const x = (p.x * scale).toFixed(6);
      const y = ((fullCanvas.height - p.y) * scale).toFixed(6);
      dxf += "10\n" + x + "\n20\n" + y + "\n";
    }
  }

  for (let a of arcs) {
    if (a.type === "poly") {
      addPoly(a.pts);
    } else if (a.type === "ellipse") {
      // approximate ellipse with N segments
      const N = 60; // more segments => smoother
      const thetaStep = (Math.PI * 2) / N;
      const pts = [];
      const rx = a.rx, ry = a.ry;
      const angleRad = (a.angle || 0) * Math.PI / 180;
      for (let i = 0; i < N; i++) {
        const t = i * thetaStep;
        let x = rx * Math.cos(t);
        let y = ry * Math.sin(t);
        // rotate by angle
        const xr = x * Math.cos(angleRad) - y * Math.sin(angleRad);
        const yr = x * Math.sin(angleRad) + y * Math.cos(angleRad);
        pts.push({ x: Math.round(a.cx + xr), y: Math.round(a.cy + yr) });
      }
      addPoly(pts);
    }
  }

  dxf += "0\nENDSEC\n0\nEOF";
  appendLog("DXF build complete.");
  return dxf;
}

// --------------------- Download helper ---------------------
function downloadFile(filename, contents, mime = "application/dxf") {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

// --------------------- UI events ---------------------
fileInput.addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    await renderFile(f);
  } catch (err) {
    appendLog("Render error: " + (err.message || err));
  }
});

detectBtn.addEventListener("click", async () => {
  try {
    appendLog("Detect pressed.");
    await detectAll();
  } catch (err) {
    appendLog("Detect error: " + (err.message || err));
  }
});

exportBtn.addEventListener("click", () => {
  try {
    if (!detected.lines || detected.lines.length === 0 && detected.circles.length === 0 && detected.arcs.length === 0) {
      appendLog("Nothing detected to export. Run Detect first.");
      return;
    }
    const dxf = buildRhinoCompatibleDXF(detected.lines, detected.circles, detected.arcs);
    downloadFile("traced_rhino_ready.dxf", dxf);
    appendLog("DXF download started.");
  } catch (err) {
    appendLog("Export error: " + (err.message || err));
  }
});

previewBtn.addEventListener("click", () => {
  appendLog("Preparing preview PNG...");
  const canvas = document.createElement("canvas");
  canvas.width = fullCanvas.width; canvas.height = fullCanvas.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(fullCanvas, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
  const url = canvas.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url; a.download = "preview_traced.png"; a.click();
  appendLog("Preview PNG downloaded.");
});

// expose readiness
window.onOpenCvReady = function() {
  appendLog("OpenCV ready.");
  if (!window.cv) appendLog("Warning: cv not present after onOpenCvReady.");
};
