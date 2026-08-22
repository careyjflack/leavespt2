const SIZE = 256;
const SCALE = 1;
const DISPLAY = SIZE * SCALE; // 256 — shrunk 50% from the original 512
const STROKE_WEIGHT = 3;

// How much padding to leave around the drawn sketch before feeding it to
// the model — matches the training data's framing (leaf fills roughly
// 45-50% width × 80% height of the frame, centered), so what the model
// sees at inference time looks like what it was trained on regardless of
// what size/position the person actually drew at.
const MODEL_INPUT_PADDING_RATIO = 0.12;

// Drawing tools
let currentTool = 'outline';           // 'outline' | 'vein'
let strokePts = [];                    // raw points captured during the current stroke
const OUTLINE_SMOOTH_WINDOW = 8;       // heavier smoothing — irons out small jagged hand tremor
const VEIN_SMOOTH_WINDOW    = 5;       // enough to kill tiny jag, without erasing the drift bulges
const CORNER_RADIUS = 6;               // px fillet applied to any sharp turn (both tools)
const CORNER_ANGLE  = 1.9;             // radians (~109°); only sharper turns round, so tips stay defined
// Organic movement is driven by Perlin noise sampled along the stroke,
// so lines stay calm most of the way and only occasionally swell or curl —
// rather than a uniform sine wobble everywhere.
const NOISE_STEP      = 0.014;   // lower = longer, rounder hills — fewer of them, but bigger
const OUTLINE_CURL    = 7.5;     // px sideways drift — bumped up to test the line having "a mind of its own" even on regular curves
const VEIN_CURL       = 10.5;    // px sideways drift for veins — a bit more soft bending
const WEIGHT_NOISE_STEP = 0.06;  // (reserved) how fast weight would breathe

// Ribbon rendering — variable-width fill instead of a flat-weight line, so
// veins can thin/thicken and every stroke end gets a soft rounded terminal.
const DENSIFY_SEG = 6;                 // spline subsamples per span — keeps ribbon edges smooth
const START_TAPER_PTS = 18;            // longer — lets the start fan out into a real web
const END_TAPER_PTS   = 10;            // longer too — gives the tip room to round into a soft blob
// A real vein junction looks thick because two consistent-width lines merge
// at a shallow angle and naturally overlap — not because of an added
// circular patch sitting on top (that reads as a bead, not a seamless
// merge). These just keep the very tip rounded, not swollen.
const OUTLINE_START_BLOB = 1.1;
const OUTLINE_END_BLOB   = 1.05;
const VEIN_START_BLOB = 1.1;
const VEIN_END_BLOB   = 1.0;
const VEIN_BASE_WEIGHT = STROKE_WEIGHT;        // veins now match the outline's weight
const VEIN_MIN_WEIGHT = VEIN_BASE_WEIGHT * 0.5;   // thinner passages
const VEIN_MAX_WEIGHT = VEIN_BASE_WEIGHT * 2.2;   // thicker passages
const VEIN_THICKNESS_NOISE_STEP = 0.022; // slower = longer, more obvious swings (not more frequent)

// Outline gets its own, gentler thickness breathing — a moderate variance,
// nowhere near as dramatic as the veins' thin/thick swings.
// Outline gets its own base weight, 1px heavier than veins, with its
// moderate thickness variance pulled in 30% (closer to a steady line).
const OUTLINE_BASE_WEIGHT = STROKE_WEIGHT + 1;
const OUTLINE_MIN_WEIGHT = OUTLINE_BASE_WEIGHT * 0.825; // was 0.75; deviation reduced 30%
const OUTLINE_MAX_WEIGHT = OUTLINE_BASE_WEIGHT * 1.245; // was 1.35; deviation reduced 30%
const OUTLINE_THICKNESS_NOISE_STEP = 0.02;

// Snapping/welding to nearby ink — outline only. On veins it was pulling
// the stroke's start/end away from where it was actually drawn and fusing
// it onto the leaf's edge, which isn't wanted; veins are left exactly as
// hand-drawn, no repositioning.
const SNAP_RADIUS = 16;          // px search radius around a stroke endpoint
const SNAP_MIN_INK_PIXELS = 6;   // require this many bright pixels before snapping (avoid false positives)
const SNAP_BLEND_PTS = 10;       // ease the snap in over this many points instead of teleporting one point
let strokeSeed = 0;              // re-randomized per stroke so none repeat
let outlineBtn, veinBtn;

let inputImg, inputCanvas, modelCanvas, output, statusMsg;
let pix2pix, transferBtn, clearBtn, modelFileInput, imageFileInput;
let isDrawing = false;
let drewThisStroke = false;
let firstSketchDone = false;
let firstOutlineStrokeDone = false; // triggers the vein-tool nudge
let firstVeinStrokeDone = false;    // triggers the generate-button nudge
let veinNudgeTooltip;
let veinNudgeTimeoutId = null;
let currentModelUrl = null;
let modelReady = false;
let modelLoadTimeoutId = null;

// Step 1: point this at the model file that should live inside the site.
// Put your .pict file in the model/ folder and keep the same relative path here.
const BUNDLED_MODEL_PATH = 'model/your-model.pict';

// ── Floyd-Steinberg dither (color) ───────────────────────────────────────────
function ditherFloydSteinbergColor(pg) {
  pg.loadPixels();
  const w = pg.width, h = pg.height;

  const r = new Float32Array(w * h);
  const g = new Float32Array(w * h);
  const b = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    r[i] = pg.pixels[o];
    g[i] = pg.pixels[o + 1];
    b[i] = pg.pixels[o + 2];
  }

  const levels = 4;
  const step = 255 / (levels - 1);
  const clamp = v => Math.min(255, Math.max(0, v));
  const quantize = v => Math.round(Math.round(v / step) * step);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const oldR = r[idx], oldG = g[idx], oldB = b[idx];
      const newR = quantize(oldR), newG = quantize(oldG), newB = quantize(oldB);
      r[idx] = newR; g[idx] = newG; b[idx] = newB;
      const errR = oldR - newR, errG = oldG - newG, errB = oldB - newB;
      if (x + 1 < w) {
        r[idx+1] = clamp(r[idx+1] + errR*7/16);
        g[idx+1] = clamp(g[idx+1] + errG*7/16);
        b[idx+1] = clamp(b[idx+1] + errB*7/16);
      }
      if (y + 1 < h) {
        if (x - 1 >= 0) {
          r[idx+w-1] = clamp(r[idx+w-1] + errR*3/16);
          g[idx+w-1] = clamp(g[idx+w-1] + errG*3/16);
          b[idx+w-1] = clamp(b[idx+w-1] + errB*3/16);
        }
        r[idx+w] = clamp(r[idx+w] + errR*5/16);
        g[idx+w] = clamp(g[idx+w] + errG*5/16);
        b[idx+w] = clamp(b[idx+w] + errB*5/16);
        if (x + 1 < w) {
          r[idx+w+1] = clamp(r[idx+w+1] + errR*1/16);
          g[idx+w+1] = clamp(g[idx+w+1] + errG*1/16);
          b[idx+w+1] = clamp(b[idx+w+1] + errB*1/16);
        }
      }
    }
  }

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    pg.pixels[o]   = r[i];
    pg.pixels[o+1] = g[i];
    pg.pixels[o+2] = b[i];
    pg.pixels[o+3] = 255;
  }
  pg.updatePixels();
}

function setup() {
  pixelDensity(1);

  inputCanvas = createCanvas(DISPLAY, DISPLAY);
  inputCanvas.class('border-box').parent('input');

  modelCanvas = createGraphics(SIZE, SIZE);
  modelCanvas.pixelDensity(1);

  background(0);

  output    = select('#output');
  statusMsg = select('#status');

  transferBtn = select('#transferBtn');
  clearBtn    = select('#clearBtn');
  clearBtn.mousePressed(clearCanvas);

  outlineBtn = select('#outlineBtn');
  veinBtn    = select('#veinBtn');
  veinNudgeTooltip = select('#veinNudgeTooltip');
  if (outlineBtn) outlineBtn.mousePressed(() => setTool('outline'));
  if (veinBtn)    veinBtn.mousePressed(() => setTool('vein'));

  modelFileInput = select('#modelFileInput');
  if (modelFileInput) modelFileInput.elt.addEventListener('change', handleModelFile);

  imageFileInput = select('#imageFileInput');
  if (imageFileInput) imageFileInput.elt.addEventListener('change', handleImageFile);

  stroke(255);
  strokeWeight(STROKE_WEIGHT);
  strokeCap(ROUND);
  strokeJoin(ROUND);

  // p5 accessibility: describe the canvas for screen readers
  describe(
    'A black 256 by 256 pixel drawing canvas. Use a mouse or touch to sketch a leaf outline and its veins using thin white lines on a black background, or upload your own image. Use the outline and vein tool icons to the left of the canvas, then click the Transfer button to generate an AI-rendered image from your sketch.'
  );

  transferBtn.mousePressed(transfer);

  // Step 2: try to load a bundled model automatically when the page opens.
  transferBtn.attribute('disabled', '');
  loadBundledModel();
}

function overCanvas() {
  return mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height;
}

function draw() {
  if (mouseIsPressed && overCanvas()) {
    isDrawing = true;
    drewThisStroke = true;

    // Capture the pointer sample only when it has actually moved a bit,
    // so we don't pile up duplicate points that make the spline lumpy.
    const p = { x: mouseX, y: mouseY };
    const last = strokePts[strokePts.length - 1];
    if (!last || dist(last.x, last.y, p.x, p.y) > 1.2) {
      strokePts.push(p);
    }

    // Re-render the current stroke fresh each frame as one continuous,
    // smoothed curve. Because we redraw only the in-progress stroke over
    // what's already committed to the canvas, finished strokes stay put.
    renderSmoothStroke(strokePts, currentTool);
  } else {
    isDrawing = false;
  }

  // Update canvas description dynamically based on drawing state
  describeElement(
    inputCanvas.elt,
    isDrawing
      ? 'Drawing in progress on the sketch canvas.'
      : 'Sketch canvas. Draw white lines, or upload your own image, then click Transfer.',
    LABEL
  );
}

let strokeSnapshot = null; // pixels committed before the current stroke began
let startAnchor = null;    // where the stroke's start snapped to, if anything

// Scan a small window of a p5.Image for bright (already-drawn) pixels and
// return their centroid — the point we should snap an endpoint onto so a
// new stroke visually welds to whatever it's meeting, instead of landing a
// few px off and leaving a gap the "web" can't bridge.
function sampleInkCentroid(img, cx, cy, radius) {
  if (!img) return null;
  img.loadPixels();
  const w = img.width, h = img.height;
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(w - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(h - 1, Math.ceil(cy + radius));
  let sx = 0, sy = 0, n = 0;
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      const idx = 4 * (yy * w + xx);
      if (img.pixels[idx] > 128) { sx += xx; sy += yy; n++; }
    }
  }
  if (n < SNAP_MIN_INK_PIXELS) return null;
  return { x: sx / n, y: sy / n };
}

// Snap the first/last point of a finished stroke onto nearby existing ink
// (checked against strokeSnapshot, which holds everything drawn BEFORE this
// stroke — so a stroke never snaps to itself).
// Snap the first/last point of a finished stroke onto nearby existing ink
// (checked against strokeSnapshot, which holds everything drawn BEFORE this
// stroke — so a stroke never snaps to itself).
//
// IMPORTANT: we don't just teleport the single endpoint to the anchor —
// that left every neighboring point exactly where it was, so if the click
// was even a few px off, point 0 would jump while point 1 stayed put,
// guaranteeing a visible kink right at the join. Instead we ease the same
// offset into the first/last several points with decaying weight, so the
// path curves smoothly toward the anchor instead of jumping to it.
function blendSnapOffset(pts, anchor, fromStart) {
  if (!anchor) return pts;
  const idx = fromStart ? 0 : pts.length - 1;
  const dx = anchor.x - pts[idx].x;
  const dy = anchor.y - pts[idx].y;
  const out = pts.slice();
  const n = Math.min(SNAP_BLEND_PTS, pts.length);
  for (let k = 0; k < n; k++) {
    const i = fromStart ? k : pts.length - 1 - k;
    const w = 1 - k / n; // 1 right at the endpoint, easing to 0 further in
    out[i] = { x: pts[i].x + dx * w, y: pts[i].y + dy * w };
  }
  return out;
}

function snapEndpointsToInk(rawPts) {
  if (rawPts.length === 0 || !strokeSnapshot) return rawPts;
  let out = rawPts;
  if (startAnchor) out = blendSnapOffset(out, startAnchor, true);
  const last = out[out.length - 1];
  const endAnchor = sampleInkCentroid(strokeSnapshot, last.x, last.y, SNAP_RADIUS);
  if (endAnchor) out = blendSnapOffset(out, endAnchor, false);
  return out;
}

function mousePressed() {
  if (overCanvas()) {
    strokePts = [];
    strokeSeed = random(1000); // fresh organic wander for each stroke
    // Snapshot everything already drawn so we can restore it each frame
    // and paint the smoothed in-progress stroke on top exactly once.
    strokeSnapshot = get();
    // Only the outline tool snaps to nearby ink — veins draw exactly where
    // the hand moved, no repositioning onto the leaf edge or anything else.
    startAnchor = currentTool === 'outline'
      ? sampleInkCentroid(strokeSnapshot, mouseX, mouseY, SNAP_RADIUS)
      : null;
  }
}

// Moving-average smoothing over the raw points — this is what makes the
// line feel buttery instead of jagged. Endpoints are preserved so the
// stroke still starts and ends where the user's pen did.
function smoothPoints(pts, radius) {
  if (pts.length <= 2) return pts.slice();
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    let sx = 0, sy = 0, n = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = i + k;
      if (j < 0 || j >= pts.length) continue;
      sx += pts[j].x; sy += pts[j].y; n++;
    }
    out.push({ x: sx / n, y: sy / n });
  }
  out[0] = pts[0];
  out[out.length - 1] = pts[pts.length - 1];
  return out;
}

// Round off any sharp direction change so the line never has a hard corner.
// At each vertex we measure the turn; if it's sharper than CORNER_ANGLE we
// replace that single point with a short arc of points that eases around the
// bend (a "fillet"). Applies to every stroke, outline and vein alike, so
// roundness is always assumed even on a fast angular scribble.
function roundCorners(pts) {
  if (pts.length < 3) return pts.slice();
  const out = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1], b = pts[i], c = pts[i + 1];
    let v1x = a.x - b.x, v1y = a.y - b.y;
    let v2x = c.x - b.x, v2y = c.y - b.y;
    const l1 = Math.hypot(v1x, v1y) || 1;
    const l2 = Math.hypot(v2x, v2y) || 1;
    const cosang = (v1x * v2x + v1y * v2y) / (l1 * l2);
    const angle = Math.acos(Math.max(-1, Math.min(1, cosang)));

    if (angle >= CORNER_ANGLE) {
      out.push(b); // already gentle, leave it
      continue;
    }
    const r = Math.min(CORNER_RADIUS, l1 * 0.5, l2 * 0.5);
    const p1 = { x: b.x + (v1x / l1) * r, y: b.y + (v1y / l1) * r };
    const p2 = { x: b.x + (v2x / l2) * r, y: b.y + (v2y / l2) * r };
    const STEPS = 5;
    for (let s = 0; s <= STEPS; s++) {
      const t = s / STEPS, mt = 1 - t;
      // quadratic Bézier p1 -> b -> p2 = a smooth arc rounding the corner
      out.push({
        x: mt * mt * p1.x + 2 * mt * t * b.x + t * t * p2.x,
        y: mt * mt * p1.y + 2 * mt * t * b.y + t * t * p2.y,
      });
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// Gentle organic bump for the vein tool: a low-frequency sine offset along
// the stroke's normal direction. Smooth and rolling, never per-pixel noise.
// Offset each point sideways by a noise value sampled along the stroke's
// arc-length. Noise gives long calm runs punctuated by occasional swells and
// curls — the "only sometimes" organic movement, applied to any tool.
function applyOrganicDrift(pts, curlAmp) {
  if (pts.length < 3) return pts.slice();
  const out = [];
  let arc = 0;
  for (let i = 0; i < pts.length; i++) {
    if (i > 0) arc += dist(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y);
    const prev = pts[Math.max(0, i - 1)];
    const next = pts[Math.min(pts.length - 1, i + 1)];
    let nx = -(next.y - prev.y);
    let ny =  (next.x - prev.x);
    const len = Math.hypot(nx, ny) || 1;
    nx /= len; ny /= len;
    // Two noise wavelengths layered: a long one bows big outlines, a shorter
    // one bends even short/straight strokes (like veins) so nothing stays
    // ruler-straight. Both centered to -1..1 so the line drifts both ways.
    const coarse = noise(strokeSeed + arc * NOISE_STEP) - 0.5;
    const fine   = noise(strokeSeed + 50 + arc * NOISE_STEP * 2.2) - 0.5;
    const n = coarse * 0.82 + fine * 0.18;
    // Ease the drift in/out over the first & last few points so the stroke
    // still lands near where the pen did, but the middle is free to bow.
    const edgeFade = Math.min(1, i / 4, (pts.length - 1 - i) / 4);
    const w = n * 2 * curlAmp * edgeFade;
    out.push({ x: pts[i].x + nx * w, y: pts[i].y + ny * w });
  }
  return out;
}

// Subdivide the point list along a Catmull-Rom spline so the ribbon we
// build next has enough resolution to look smooth-edged rather than faceted.
function densify(pts, segPerSpan) {
  if (pts.length < 3) return pts.slice();
  const dense = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let s = 0; s < segPerSpan; s++) {
      const t = s / segPerSpan;
      dense.push({
        x: curvePoint(p0.x, p1.x, p2.x, p3.x, t),
        y: curvePoint(p0.y, p1.y, p2.y, p3.y, t),
      });
    }
  }
  dense.push(pts[pts.length - 1]);
  return dense;
}

// Draw the stroke as ONE continuous filled polygon: a ribbon whose width can
// vary along its length, tapering toward a rounded terminal at each end that's
// part of the same shape (not a separate overlapping circle) — so it merges
// smoothly into the line instead of leaving a seam or hook.
//
// The two ends behave differently on purpose: strokes are usually drawn
// STARTING at a junction (a vein leaving the midrib or outline) and ENDING
// at a free tip out in the leaf body. So the start fans out into a webbed
// join, while the end eases down to a fine point — like a real vein.
function drawRibbonStroke(pts, thicknessAt, baseWeight, startBlobScale, endBlobScale, startTaperPts, endTaperPts) {
  const N = pts.length;
  const normals = [];
  const widths = [];
  let arc = 0;

  for (let i = 0; i < N; i++) {
    if (i > 0) arc += dist(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y);
    const prev = pts[Math.max(0, i - 1)];
    const next = pts[Math.min(N - 1, i + 1)];
    let nx = -(next.y - prev.y);
    let ny =  (next.x - prev.x);
    const len = Math.hypot(nx, ny) || 1;
    nx /= len; ny /= len;
    normals.push({ x: nx, y: ny });

    // Base thickness, then ease toward a controlled terminal size near each
    // end — blended in (not multiplied), so a naturally-thick spot in the
    // noise can't compound into an oversized ball at the tip.
    let w = thicknessAt(arc);
    const distFromStart = i;
    const distFromEnd = N - 1 - i;
    if (distFromStart < startTaperPts) {
      const t = 1 - distFromStart / startTaperPts;
      const target = baseWeight * startBlobScale;
      w = w + (target - w) * (t * t);
    } else if (distFromEnd < endTaperPts) {
      const t = 1 - distFromEnd / endTaperPts;
      const target = baseWeight * endBlobScale;
      w = w + (target - w) * (t * t);
    }
    widths.push(w);
  }

  const left = pts.map((p, i) => ({ x: p.x + normals[i].x * widths[i] / 2, y: p.y + normals[i].y * widths[i] / 2 }));
  const right = pts.map((p, i) => ({ x: p.x - normals[i].x * widths[i] / 2, y: p.y - normals[i].y * widths[i] / 2 }));

  // Round caps: a soft semicircle at each end, using the SAME radius as the
  // ribbon's own tapered end width, so it's continuous with no seam.
  const CAP_STEPS = 12;
  const startCap = [];
  const startAngle = Math.atan2(normals[0].y, normals[0].x);
  const startR = widths[0] / 2;
  for (let s = 1; s < CAP_STEPS; s++) {
    const a = startAngle + (s / CAP_STEPS) * Math.PI;
    startCap.push({ x: pts[0].x + Math.cos(a) * startR, y: pts[0].y + Math.sin(a) * startR });
  }
  const endCap = [];
  const endAngle = Math.atan2(normals[N - 1].y, normals[N - 1].x);
  const endR = widths[N - 1] / 2;
  for (let s = 1; s < CAP_STEPS; s++) {
    const a = endAngle - (s / CAP_STEPS) * Math.PI;
    endCap.push({ x: pts[N - 1].x + Math.cos(a) * endR, y: pts[N - 1].y + Math.sin(a) * endR });
  }

  noStroke();
  fill(255);
  beginShape();
  for (const p of left) vertex(p.x, p.y);
  for (const p of endCap) vertex(p.x, p.y);
  for (let i = right.length - 1; i >= 0; i--) vertex(right[i].x, right[i].y);
  for (const p of startCap) vertex(p.x, p.y);
  endShape(CLOSE);
}

function renderSmoothStroke(rawPts, tool) {
  if (strokeSnapshot) image(strokeSnapshot, 0, 0);
  if (rawPts.length === 0) return;

  const smoothWin = tool === 'vein' ? VEIN_SMOOTH_WINDOW : OUTLINE_SMOOTH_WINDOW;
  let pts = smoothPoints(rawPts, smoothWin);
  pts = roundCorners(pts);           // soften any sharp turn — both tools
  const curl = tool === 'vein' ? VEIN_CURL : OUTLINE_CURL;
  pts = applyOrganicDrift(pts, curl);

  if (pts.length === 1) {
    noStroke();
    fill(255);
    ellipse(pts[0].x, pts[0].y, VEIN_BASE_WEIGHT * VEIN_START_BLOB, VEIN_BASE_WEIGHT * VEIN_START_BLOB);
    return;
  }
  if (pts.length === 2) {
    stroke(255);
    strokeWeight(STROKE_WEIGHT);
    strokeCap(ROUND);
    line(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
    return;
  }

  pts = densify(pts, DENSIFY_SEG);

  if (tool === 'vein') {
    // Thickness wanders slowly within a bounded range, matching how much
    // variation the reference actually shows — a little thin/thick breathing,
    // not wild jitter.
    const thicknessAt = (arc) => {
      const n = noise(strokeSeed + 300 + arc * VEIN_THICKNESS_NOISE_STEP);
      return lerp(VEIN_MIN_WEIGHT, VEIN_MAX_WEIGHT, n);
    };
    drawRibbonStroke(pts, thicknessAt, VEIN_BASE_WEIGHT, VEIN_START_BLOB, VEIN_END_BLOB, START_TAPER_PTS, END_TAPER_PTS);
  } else {
    // Outline gets its own gentler thickness breathing — moderate variance,
    // not the dramatic swings the veins have.
    const thicknessAt = (arc) => {
      const n = noise(strokeSeed + 300 + arc * OUTLINE_THICKNESS_NOISE_STEP);
      return lerp(OUTLINE_MIN_WEIGHT, OUTLINE_MAX_WEIGHT, n);
    };
    drawRibbonStroke(pts, thicknessAt, OUTLINE_BASE_WEIGHT, OUTLINE_START_BLOB, OUTLINE_END_BLOB, START_TAPER_PTS, END_TAPER_PTS);
  }
}

function showVeinNudge() {
  if (veinBtn && veinBtn.elt) veinBtn.elt.classList.add('tool-nudge');
  if (veinNudgeTooltip && veinNudgeTooltip.elt) veinNudgeTooltip.elt.hidden = false;
  // don't nudge forever if they don't notice
  clearTimeout(veinNudgeTimeoutId);
  veinNudgeTimeoutId = setTimeout(hideVeinNudge, 9000);
}

function hideVeinNudge() {
  clearTimeout(veinNudgeTimeoutId);
  if (veinBtn && veinBtn.elt) veinBtn.elt.classList.remove('tool-nudge');
  if (veinNudgeTooltip && veinNudgeTooltip.elt) veinNudgeTooltip.elt.hidden = true;
}

function setTool(tool) {
  currentTool = tool;
  if (outlineBtn) {
    outlineBtn.elt.classList.toggle('is-active', tool === 'outline');
    outlineBtn.attribute('aria-pressed', tool === 'outline' ? 'true' : 'false');
  }
  if (veinBtn) {
    veinBtn.elt.classList.toggle('is-active', tool === 'vein');
    veinBtn.attribute('aria-pressed', tool === 'vein' ? 'true' : 'false');
  }
  // Switching to the vein tool satisfies the nudge — dismiss it.
  if (tool === 'vein') hideVeinNudge();
}

function mouseReleased() {
  // Only the outline gets a final corrected pass that welds its start/end
  // onto nearby ink. Veins are left exactly as drawn — no repositioning.
  if (strokePts.length > 0 && strokeSnapshot && currentTool === 'outline') {
    const snapped = snapEndpointsToInk(strokePts);
    renderSmoothStroke(snapped, currentTool);
  }
  strokeSnapshot = null;
  startAnchor = null;
  strokePts = [];

  if (drewThisStroke) {
    firstSketchDone = true;
    if (currentTool === 'outline' && !firstOutlineStrokeDone) {
      // First stroke ever, drawn as an outline — point them at the vein tool next.
      firstOutlineStrokeDone = true;
      showVeinNudge();
    } else if (currentTool === 'vein' && !firstVeinStrokeDone) {
      // First vein drawn — now nudge them toward generating.
      firstVeinStrokeDone = true;
      hideVeinNudge();
      if (transferBtn && transferBtn.elt) {
        transferBtn.elt.classList.add('nudge');
        setTimeout(() => {
          if (transferBtn && transferBtn.elt) {
            transferBtn.elt.classList.remove('nudge');
          }
        }, 6000);
      }
    }
  }
  drewThisStroke = false;
}

function beginModelLoad(message) {
  clearTimeout(modelLoadTimeoutId);
  modelReady = false;
  transferBtn.attribute('disabled', '');
  statusMsg.html(message);

  modelLoadTimeoutId = setTimeout(() => {
    if (!modelReady) {
      statusMsg.html('');
    }
  }, 8000);
}

function loadBundledModel() {
  // ml5 comes from a CDN; if it didn't load, say so clearly instead of hanging.
  if (typeof ml5 === 'undefined') {
    statusMsg.html('Could not reach the ml5 library. Check your internet connection and reload.');
    return;
  }

  beginModelLoad('');

  if (currentModelUrl) {
    URL.revokeObjectURL(currentModelUrl);
    currentModelUrl = null;
  }

  pix2pix = ml5.pix2pix(BUNDLED_MODEL_PATH, modelLoaded);
}

function handleModelFile(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  // Clean up any previously created object URL.
  if (currentModelUrl) {
    URL.revokeObjectURL(currentModelUrl);
    currentModelUrl = null;
  }

  beginModelLoad('Loading model... Please wait...');

  currentModelUrl = URL.createObjectURL(file);
  pix2pix = ml5.pix2pix(currentModelUrl, modelLoaded);
}

function handleImageFile(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  loadImage(url, img => {
    background(0);
    image(img, 0, 0, DISPLAY, DISPLAY);
    inputImg = img;
    URL.revokeObjectURL(url);
  }, err => {
    console.log(err);
    statusMsg.html('Could not load that image file.');
    URL.revokeObjectURL(url);
  });
}

function modelLoaded() {
  clearTimeout(modelLoadTimeoutId);
  modelReady = true;
  statusMsg.html('Model Loaded!');
  transferBtn.elt.removeAttribute('disabled');
}

function clearCanvas() {
  statusMsg.html(modelReady ? 'Model loaded — draw a leaf and hit Transfer.' : 'Loading the model…');
  background(0);
  strokePts = [];
  strokeSnapshot = null;
  startAnchor = null;
  output.elt.src = 'images/blank.png';
  output.elt.alt = 'The AI-generated output will appear here after clicking Transfer.';
}

// Find the bounding box of everything drawn (bright pixels) on a p5.Image.
function findInkBoundingBox(img) {
  img.loadPixels();
  const w = img.width, h = img.height;
  let minX = w, maxX = -1, minY = h, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (img.pixels[4 * (y * w + x)] > 40) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // nothing drawn
  return { minX, maxX, minY, maxY };
}

// Build the square region to feed the model: a tight crop around whatever
// was drawn, padded the same way the training data is framed, so the
// leaf's scale/position in what the model sees matches what it learned on
// — regardless of how big or where the person actually drew it.
function buildNormalizedModelInput(sourceCanvas) {
  const box = findInkBoundingBox(sourceCanvas);
  const full = { x: 0, y: 0, w: sourceCanvas.width, h: sourceCanvas.height };
  if (!box) return full; // nothing drawn yet — fall back to the whole canvas

  const bw = box.maxX - box.minX;
  const bh = box.maxY - box.minY;
  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  const side = Math.max(bw, bh) * (1 + MODEL_INPUT_PADDING_RATIO * 2);

  let left = cx - side / 2;
  let top = cy - side / 2;
  // Keep the crop on-canvas rather than sampling outside it.
  left = Math.max(0, Math.min(left, sourceCanvas.width - side));
  top = Math.max(0, Math.min(top, sourceCanvas.height - side));
  return { x: left, y: top, w: side, h: side };
}

function transfer() {
  if (!pix2pix || !modelReady) {
    statusMsg.html('The model is not ready yet — give it a moment to load.');
    return;
  }

  statusMsg.html('Transferring...');
  output.elt.alt = 'Generating AI image from your sketch, please wait.';

  const sketchSnapshot = get();
  const region = buildNormalizedModelInput(sketchSnapshot);
  modelCanvas.image(
    sketchSnapshot,
    0, 0, SIZE, SIZE,                 // destination: fill the model's full frame
    region.x, region.y, region.w, region.h  // source: the normalized, dataset-matching crop
  );

  pix2pix.transfer(modelCanvas.elt, function(err, result) {
    if (err) { console.log(err); return; }
    if (result && result.src) {
      statusMsg.html('generation done!');

      loadImage(result.src, p5img => {
        const tmp = createGraphics(DISPLAY, DISPLAY);
        tmp.pixelDensity(1);
        tmp.image(p5img, 0, 0, DISPLAY, DISPLAY);
        ditherFloydSteinbergColor(tmp);

        tmp.canvas.toBlob(blob => {
          output.elt.src = URL.createObjectURL(blob);
          output.elt.alt = 'AI-generated image produced from your line drawing. A color-dithered image based on the sketch you drew.';
        });
      });
    }
  });
}