// EdgeDefectNet web app: drag-and-drop / example input, in-browser ONNX inference, animated results, 3D wafer.
// Class names, colours and descriptions come from classes.json, generated from edgedefectnet/classes.py.

import { predict, toModelCanvas, warmRuntime } from "./inference.js";
import { createWafer } from "./wafer3d.js";

const MAX_FILE_BYTES = 25 * 1024 * 1024; // same limit as the Python package
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

/** Tiny DOM builder: h("div", {class: "x"}, "text", child). No HTML strings, so data can never inject markup. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.setAttribute("style", v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
const svg = (tag, attrs = {}) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

let classes = [];
let wafer = null;
let runId = 0;

// ---------------- Theme ----------------

function setupTheme() {
  const btn = $("theme");
  const apply = (t, persist) => {
    document.documentElement.dataset.theme = t;
    btn.setAttribute("aria-pressed", String(t === "light"));
    btn.setAttribute("aria-label", t === "light" ? "Switch to dark theme" : "Switch to light theme");
    document.querySelector('meta[name="theme-color"]').content = t === "light" ? "#f3f2ff" : "#0b0a14";
    if (persist) {
      try { localStorage.setItem("edn-theme", t); } catch { /* storage blocked */ }
    }
  };
  apply(document.documentElement.dataset.theme, false);
  btn.addEventListener("click", () => apply(document.documentElement.dataset.theme === "light" ? "dark" : "light", true));
}

// ---------------- Class glyphs: schematic wafer maps of each class's typical pattern ----------------

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function failPattern(name, x, y, rand) {
  // x, y in [-1, 1]; returns true when this die is drawn as failing
  const r = Math.hypot(x, y);
  switch (name) {
    case "Bridges": return rand() < 0.86;
    case "Clean": return rand() < 0.04;
    case "Cracks": return Math.abs(r - 0.62) < 0.16 && rand() < 0.9; // donut
    case "LER": return r > 0.78 && rand() < 0.92; // edge ring
    case "Opens": return Math.hypot(x - 0.38, y + 0.3) < 0.36 && rand() < 0.9; // local cluster
    case "Other": return rand() < 0.2;
    case "Scratches": return Math.abs(y - (0.9 * x - 0.1) ) < 0.1 && rand() < 0.95; // streak
    case "Shorts": return r < 0.4 && rand() < 0.95; // centre
    case "Vias": return (Math.round((x + 1) * 3.5) % 2 === 0) && (Math.round((y + 1) * 3.5) % 2 === 0); // regular array
    default: return false;
  }
}

function glyph(name, size = 64) {
  const n = 13;
  const rand = rng(name.length * 977 + name.charCodeAt(0));
  const el = svg("svg", { class: "glyph", viewBox: "-1.1 -1.1 2.2 2.2", width: size, height: size, "aria-hidden": "true" });
  el.append(svg("circle", { class: "rim", r: "1.04", "stroke-width": "0.035" }));
  const step = 2 / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = -1 + step * (i + 0.5), y = -1 + step * (j + 0.5);
      if (Math.hypot(x, y) > 0.96) continue;
      const bad = failPattern(name, x, y, rand);
      el.append(svg("circle", { class: bad ? "bad" : "ok", cx: x.toFixed(3), cy: y.toFixed(3), r: (step * 0.36).toFixed(3) }));
    }
  }
  return el;
}

// ---------------- UI builders ----------------

function buildLegend() {
  const ul = $("legend");
  classes.forEach((c, i) => {
    const b = h("button", { type: "button", style: `--c:${c.color}`, "data-i": i }, h("i"), c.name);
    b.addEventListener("pointerenter", () => highlight(i));
    b.addEventListener("pointerleave", () => highlight(-1));
    b.addEventListener("focus", () => highlight(i));
    b.addEventListener("blur", () => highlight(-1));
    ul.append(h("li", {}, b));
  });
}

function highlight(i) {
  wafer?.highlight(i);
  document.querySelectorAll("#legend button").forEach((b) => b.classList.toggle("on", Number(b.dataset.i) === i));
  const sub = $("stageSub");
  if (i >= 0) {
    sub.textContent = classes[i].summary;
  } else {
    sub.textContent = lastPrediction ? `Predicted ${lastPrediction}. Arc brightness follows the probabilities.` : "Arc brightness follows the model's probabilities";
  }
}

function buildClassCards() {
  const grid = $("classGrid");
  classes.forEach((c, i) => {
    const card = h(
      "article",
      { class: "class-card", style: `--c:${c.color}`, tabindex: "0", "data-name": c.name, "aria-label": `${c.name}: ${c.summary}` },
      h("header", {}, glyph(c.name, 64), h("h3", {}, c.name), h("span", { class: "tag", hidden: true }, "Predicted")),
      h("p", { class: "summary" }, c.summary),
      h("p", {}, h("b", {}, "In this dataset: "), c.looks_like),
      h("p", { class: "src" }, c.source)
    );
    card.querySelector(".tag").hidden = true;
    card.addEventListener("pointerenter", () => highlight(i));
    card.addEventListener("pointerleave", () => highlight(-1));
    card.addEventListener("focus", () => highlight(i));
    card.addEventListener("blur", () => highlight(-1));
    grid.append(card);
  });
}

async function buildExamples() {
  const box = $("examples");
  let list = [];
  try {
    list = await (await fetch("examples/index.json")).json();
  } catch {
    box.replaceChildren(h("p", { class: "hint" }, "Examples are unavailable."));
    return;
  }
  for (const file of list) {
    const truth = file.split("__")[0];
    const b = h("button", { class: "example", type: "button", "aria-pressed": "false", "data-truth": truth },
      h("img", { src: `examples/${file}`, alt: `Example wafer image, true class ${truth}`, loading: "lazy" }), h("span", {}, truth));
    b.addEventListener("click", async () => {
      document.querySelectorAll(".example").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      try {
        const img = await loadImage(`examples/${file}`);
        showPreview(img, file, `${img.naturalWidth}×${img.naturalHeight}px example`);
        await classify(img, truth);
      } catch (err) {
        fail(`Couldn't load the example: ${err.message}`);
      }
    });
    box.append(b);
  }
}

// ---------------- Result panel ----------------

let lastPrediction = null;
let lastRetry = null;

function setState(state) {
  $("result").dataset.state = state;
}

function fail(message, retry = null) {
  setState("error");
  $("errorText").textContent = message;
  lastRetry = retry;
  $("retry").hidden = !retry;
}

function busy(text, fraction = null, hint = "") {
  setState("busy");
  $("busyText").textContent = text;
  $("busyHint").textContent = hint;
  const p = $("progress");
  p.classList.toggle("indeterminate", fraction == null);
  const pct = fraction == null ? 0 : Math.round(fraction * 100);
  p.setAttribute("aria-valuenow", String(pct));
  $("bar").style.width = fraction == null ? "" : `${pct}%`;
}

function onStage(s) {
  if (s.stage === "runtime") busy("Loading ONNX Runtime…");
  else if (s.stage === "download") {
    const mb = (s.bytes / 1e6).toFixed(0);
    busy("Downloading the model…", s.fraction, s.fraction == null ? "" : `${mb} of about 45 MB. This happens once; the browser caches it.`);
  } else if (s.stage === "start") busy("Starting the model…");
}

function countUp(el, to, ms = 900) {
  if (reduceMotion.matches) { el.textContent = Math.round(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const CIRC = 2 * Math.PI * 68;

function render(probs, truth, ms, firstRun) {
  const ranked = classes.map((c, i) => ({ c, i, p: probs[i] })).sort((a, b) => b.p - a.p);
  const [best, second] = ranked;
  lastPrediction = best.c.name;
  setState("done");

  const result = $("result");
  result.style.setProperty("--c", best.c.color);
  $("topLabel").textContent = best.c.name;
  $("topConf").textContent = `${(best.p * 100).toFixed(1)}% confidence`;
  $("truth").textContent = truth ? (truth === best.c.name ? `Matches the true class (${truth}).` : `True class: ${truth}.`) : "";

  // radial gauge
  const gauge = result.querySelector(".gauge");
  gauge.style.setProperty("--c", best.c.color);
  for (const id of ["gArc", "gGlow"]) {
    const arc = $(id);
    arc.style.transition = "none";
    arc.style.strokeDasharray = `${CIRC} ${CIRC}`;
    arc.style.strokeDashoffset = String(CIRC);
    arc.getBoundingClientRect(); // restart the transition
    arc.style.transition = "";
    requestAnimationFrame(() => (arc.style.strokeDashoffset = String(CIRC * (1 - best.p))));
  }
  countUp($("gNum"), best.p * 100);

  const note = $("confNote");
  const low = best.p < 0.6;
  note.classList.toggle("low", low);
  note.textContent = low
    ? `Low confidence: ${best.c.name} (${(best.p * 100).toFixed(0)}%) is only slightly ahead of ${second.c.name} (${(second.p * 100).toFixed(0)}%). Treat this as uncertain.`
    : "The score is a softmax probability, not a calibrated one, and the model always picks one of the nine classes.";

  // ranked bars
  const bars = $("bars");
  bars.replaceChildren();
  for (const { c, p } of ranked) {
    const fill = h("span", { class: "fill" });
    bars.append(h("li", { style: `--c:${c.color}` }, h("span", { class: "name" }, c.name), h("span", { class: "track" }, fill), h("span", { class: "pct" }, `${(p * 100).toFixed(1)}%`)));
    requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = `${Math.max(p * 100, 0.6)}%`)));
  }

  // explainer for the predicted class
  $("explain").replaceChildren(
    glyph(best.c.name, 76),
    h("div", {}, h("h4", {}, `About ${best.c.name}`), h("p", {}, best.c.summary), h("p", {}, h("b", {}, "In this dataset: "), best.c.looks_like), h("p", {}, best.c.source))
  );
  $("explain").style.setProperty("--c", best.c.color);

  $("timing").textContent = `Ran on this device in ${ms.toFixed(0)} ms${firstRun ? " (the model started up just now)" : ""}. Nothing was uploaded.`;

  document.querySelectorAll(".class-card").forEach((card) => {
    const on = card.dataset.name === best.c.name;
    card.classList.toggle("predicted", on);
    card.querySelector(".tag").hidden = !on;
  });
  document.body.dataset.predicted = best.c.name; // read by the automated browser test
}

let modelStarted = false;

async function classify(img, truth) {
  const id = ++runId;
  const canvas = toModelCanvas(img);
  wafer?.setImage(canvas);
  wafer?.setProbabilities(null);
  lastPrediction = null;
  const firstRun = !modelStarted;
  const retry = () => classify(img, truth);
  try {
    if (firstRun) busy("Starting…");
    const { probs, ms } = await predict(canvas, firstRun ? onStage : undefined);
    modelStarted = true;
    if (id !== runId) return; // a newer image was chosen meanwhile
    render(probs, truth, ms, firstRun);
    wafer?.setProbabilities(probs);
    wafer?.expose();
    highlight(-1);
  } catch (err) {
    if (id !== runId) return;
    fail(`Couldn't run the model: ${err.message}. Check your connection and try again.`, retry);
  }
}

// ---------------- Inputs ----------------

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("that file isn't an image the browser can read"));
    img.src = src;
  });
}

function showPreview(img, name, meta) {
  const c = $("previewCanvas");
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  $("previewName").textContent = name;
  $("previewMeta").textContent = meta;
  $("preview").hidden = false;
}

async function useFile(file) {
  if (!file) return;
  document.querySelectorAll(".example").forEach((b) => b.setAttribute("aria-pressed", "false"));
  if (!file.type.startsWith("image/")) return fail(`"${file.name}" isn't an image. Choose a PNG or JPG.`);
  if (file.size === 0) return fail(`"${file.name}" is empty.`);
  if (file.size > MAX_FILE_BYTES) return fail(`"${file.name}" is ${(file.size / 1e6).toFixed(0)} MB; the limit is ${MAX_FILE_BYTES / 1e6} MB.`);
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    if (Math.min(img.naturalWidth, img.naturalHeight) < 8) throw new Error("the image is smaller than 8 pixels");
    showPreview(img, file.name, `${img.naturalWidth}×${img.naturalHeight}px, ${(file.size / 1024).toFixed(0)} KB`);
    await classify(img, null);
  } catch (err) {
    fail(`Couldn't open the image: ${err.message}.`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function setupInputs() {
  const drop = $("drop"), input = $("file"), veil = $("veil");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    useFile(input.files[0]);
    input.value = ""; // choosing the same file again should still fire
  });
  drop.addEventListener("pointerenter", () => warmRuntime().catch(() => {}), { once: true });

  // dropping anywhere on the page works; the drop zone just highlights
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
  addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; veil.classList.add("on"); } });
  addEventListener("dragleave", (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; veil.classList.remove("on"); } });
  addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    veil.classList.remove("on");
    drop.classList.remove("over");
    $("try").scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth" });
    useFile(e.dataTransfer.files[0]);
  });
  drop.addEventListener("dragenter", () => drop.classList.add("over"));
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));

  addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
    if (item) useFile(item.getAsFile());
  });
  $("retry").addEventListener("click", () => lastRetry?.());
}

// ---------------- Model card ----------------

async function setupModelCard() {
  let m;
  try {
    const resp = await fetch("metrics.json");
    if (!resp.ok) throw new Error(String(resp.status));
    m = await resp.json();
  } catch {
    $("stats").replaceChildren(h("p", { class: "hint" }, "Metrics file not found."));
    return;
  }
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  $("heroAcc").textContent = pct(m.accuracy);
  $("stats").replaceChildren(
    ...[[pct(m.accuracy), "accuracy"], [m.macro_f1.toFixed(3), "macro-F1"], [m.n.toLocaleString(), "test images"]].map(([v, l]) =>
      h("div", { class: "stat" }, h("b", {}, v), h("span", {}, l)))
  );

  const names = { MULTI_CLASS: "Roboflow wafer maps", WM811K_IMG: "WM-811K 32×32 maps", IESA: "IESA challenge images" };
  $("sources").replaceChildren(
    ...Object.entries(m.per_source_accuracy).map(([s, v]) => {
      const fill = h("span", { class: "fill" });
      requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = `${v.accuracy * 100}%`)));
      return h("div", { class: "source" }, h("span", {}, names[s] || s, " ", h("small", {}, `${v.n} images`)), h("span", {}, pct(v.accuracy)), h("span", { class: "track" }, fill));
    })
  );

  // per-class table
  const table = $("perClass");
  const head = h("thead", {}, h("tr", {}, ...["Class", "Precision", "Recall", "F1", "Images"].map((t) => h("th", { scope: "col" }, t))));
  const body = h("tbody", {}, ...classes.map((c) => {
    const r = m.per_class[c.name];
    return h("tr", {}, h("td", { style: `--c:${c.color}` }, h("span", { class: "swatch" }), c.name), h("td", {}, r.precision.toFixed(3)), h("td", {}, r.recall.toFixed(3)), h("td", {}, r["f1-score"].toFixed(3)), h("td", {}, r.support));
  }));
  table.append(head, body);

  // confusion matrix: brightness follows the row-normalised count
  const cm = m.confusion_matrix, labels = m.labels, n = labels.length;
  const grid = $("matrix");
  grid.style.gridTemplateColumns = `auto repeat(${n}, minmax(28px, 1fr))`;
  const cells = [h("div", { role: "presentation" })];
  labels.forEach((l) => cells.push(h("div", { class: "lab col", role: "columnheader" }, l)));
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    cells.push(h("div", { class: "lab", role: "rowheader" }, labels[i]));
    row.forEach((v, j) => {
      const f = v / total;
      const bg = i === j
        ? `rgba(61, 224, 255, ${0.1 + f * 0.8})`
        : v ? `rgba(255, 110, 210, ${0.2 + f * 0.8})` : "transparent";
      const ink = i === j && f > 0.5 ? "#04202a" : v ? "var(--ink)" : "var(--muted)";
      cells.push(h("div", { class: "cell", role: "cell", style: `--cell-bg:${bg};--cell-ink:${ink}`, title: `True ${labels[i]}, predicted ${labels[j]}: ${v}`, "aria-label": `True ${labels[i]}, predicted ${labels[j]}: ${v}` }, v || ""));
    });
  });
  grid.replaceChildren(...cells);
}

// ---------------- Boot ----------------

async function boot() {
  setupTheme();
  setupInputs();
  try {
    const resp = await fetch("classes.json");
    if (!resp.ok) throw new Error(String(resp.status));
    classes = await resp.json();
  } catch (err) {
    fail(`Couldn't load the class list: ${err.message}. Reload the page to try again.`);
    return;
  }
  buildLegend();
  buildClassCards();
  wafer = createWafer({
    canvas: $("wafer"),
    stage: $("stage"),
    classes,
    onNote: (msg) => {
      $("stageNote").textContent = msg;
      $("stageNote").hidden = !msg;
    },
  });
  await Promise.all([buildExamples(), setupModelCard()]);
}

boot();
