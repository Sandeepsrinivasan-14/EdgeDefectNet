// EdgeDefectNet web app: ONNX inference in the browser + a 3D wafer that wears the selected image.
// Data contract with the Python package:
//   - CLASS_NAMES must match edgedefectnet/classes.py (alphabetical, = ImageFolder order = logit order)
//   - preprocessing must match edgedefectnet/preprocessing.py (L = 0.299R+0.587G+0.114B, 224x224, (x/255-0.45)/0.23)

const CLASS_NAMES = ["Bridges", "Clean", "Cracks", "LER", "Opens", "Other", "Scratches", "Shorts", "Vias"];
const SIZE = 224, MEAN = 0.45, STD = 0.23;
const ORT_CDNS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
  "https://unpkg.com/onnxruntime-web@1.20.1/dist/",
];
const THREE_CDNS = [
  "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

function loadScript(urls) {
  return new Promise((resolve, reject) => {
    const tryNext = (i) => {
      if (i >= urls.length) return reject(new Error("could not load " + urls[0]));
      const el = document.createElement("script");
      el.src = urls[i];
      el.onload = () => resolve(urls[i]);
      el.onerror = () => { el.remove(); tryNext(i + 1); };
      document.head.appendChild(el);
    };
    tryNext(0);
  });
}

// ---------------- Inference ----------------

let sessionPromise = null;

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model download failed (HTTP ${resp.status})`);
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out;
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ortUrl = await loadScript(ORT_CDNS.map((b) => b + "ort.wasm.min.js"));
      const ort = window.ort;
      ort.env.wasm.wasmPaths = ortUrl.replace(/ort\.wasm\.min\.js$/, "");
      ort.env.wasm.numThreads = 1;
      $("status").textContent = "Downloading the model…";
      $("progress").hidden = false;
      const bytes = await fetchWithProgress("model.onnx", (f) => ($("bar").style.width = `${(f * 100).toFixed(0)}%`));
      $("status").textContent = "Starting the model…";
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      $("progress").hidden = true;
      return { ort, session };
    })().catch((err) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

function toGrayscaleCanvas(img) {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  return c;
}

function toTensorData(canvas) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, SIZE, SIZE);
  const out = new Float32Array(SIZE * SIZE);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const l = Math.round((data[p] * 299 + data[p + 1] * 587 + data[p + 2] * 114) / 1000); // PIL "L"
    out[i] = (l / 255 - MEAN) / STD;
  }
  return out;
}

function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function classify(img, truth) {
  const canvas = toGrayscaleCanvas(img);
  wafer.setImage(canvas);
  $("top1").hidden = true;
  $("timing").textContent = "";
  try {
    const { ort, session } = await getSession();
    $("status").textContent = "Classifying…";
    const t0 = performance.now();
    const input = new ort.Tensor("float32", toTensorData(canvas), [1, 1, SIZE, SIZE]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const probs = softmax(Array.from(out[session.outputNames[0]].data));
    const ms = performance.now() - t0;
    render(probs, truth, ms);
    wafer.expose();
  } catch (err) {
    $("progress").hidden = true;
    $("status").textContent = `Couldn't run the model: ${err.message}. Check your connection and try again.`;
  }
}

function render(probs, truth, ms) {
  const ranked = CLASS_NAMES.map((name, i) => ({ name, p: probs[i] })).sort((a, b) => b.p - a.p);
  const best = ranked[0];
  $("status").textContent = "";
  $("top1").hidden = false;
  $("topLabel").textContent = best.name;
  $("topConf").textContent = `${(best.p * 100).toFixed(1)}% confidence`;
  $("truth").textContent = truth
    ? (truth === best.name ? `Matches the true class (${truth}).` : `True class: ${truth}.`)
    : "";
  const bars = $("bars");
  bars.innerHTML = "";
  for (const { name, p } of ranked) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${name}</span><span class="track"><span class="fill"></span></span><span class="pct">${(p * 100).toFixed(1)}%</span>`;
    bars.appendChild(li);
    requestAnimationFrame(() => (li.querySelector(".fill").style.width = `${Math.max(p * 100, 0.6)}%`));
  }
  $("timing").textContent = `Ran on this device in ${ms.toFixed(0)} ms.`;
  document.body.dataset.predicted = best.name; // used by the automated browser test
}

// ---------------- 3D wafer ----------------

const wafer = (() => {
  const api = { setImage() {}, expose() {} };
  const canvasEl = $("wafer");
  let pendingImage = null;
  api.setImage = (c) => (pendingImage = c);

  loadScript(THREE_CDNS).then(() => {
    const THREE = window.THREE;
    const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    camera.position.set(0, 4.2, 8.6);
    camera.lookAt(0, -0.2, 0);

    const blank = document.createElement("canvas");
    blank.width = blank.height = 8;
    const tex = new THREE.CanvasTexture(blank);

    const uniforms = {
      uMap: { value: tex }, uHasMap: { value: 0 }, uTime: { value: 0 }, uScan: { value: -2 },
      uSafelight: { value: new THREE.Color("#FFC53D") },
    };
    const faceMat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: `
        varying vec2 vUv; varying vec3 vNormalW; varying vec3 vPosW;
        void main() {
          vUv = uv;
          vec4 w = modelMatrix * vec4(position, 1.0);
          vPosW = w.xyz; vNormalW = normalize(mat3(modelMatrix) * normal);
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `
        uniform sampler2D uMap; uniform float uHasMap; uniform float uTime; uniform float uScan; uniform vec3 uSafelight;
        varying vec2 vUv; varying vec3 vNormalW; varying vec3 vPosW;
        vec3 film(float t) { return 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + t)); }
        void main() {
          vec2 p = vUv * 2.0 - 1.0;
          float r = length(p);
          if (r > 1.0) discard;
          vec3 V = normalize(cameraPosition - vPosW);
          float cosT = clamp(dot(normalize(vNormalW), V), 0.0, 1.0);
          // thin-film interference: hue shifts with viewing angle and radius, like a real polished wafer
          vec3 irid = film(1.6 * cosT + 0.55 * r + 0.04 * sin(uTime * 0.4));
          vec3 base = vec3(0.62, 0.63, 0.70);  // polished silicon
          float img = texture2D(uMap, vec2(vUv.x, 1.0 - vUv.y)).r;
          vec3 surface = mix(base, vec3(0.08) + vec3(img) * 0.9, uHasMap);
          vec3 col = mix(surface, surface * (0.55 + 0.75 * irid), mix(0.45, 0.30, uHasMap));
          // die grid
          vec2 g = abs(fract(vUv * 26.0) - 0.5);
          float grid = smoothstep(0.47, 0.5, max(g.x, g.y));
          col = mix(col, col * 0.55, grid * 0.6);
          // specular sheen
          col += pow(1.0 - cosT, 3.0) * 0.25;
          // exposure scan line
          float d = abs(vUv.y - uScan);
          col += uSafelight * smoothstep(0.08, 0.0, d) * 0.9;
          // edge exclusion ring
          col = mix(col, col * 0.65, smoothstep(0.93, 0.99, r));
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
    const edgeMat = new THREE.MeshStandardMaterial({ color: 0x8a8aa0, metalness: 0.9, roughness: 0.3 });
    const geo = new THREE.CylinderGeometry(2.4, 2.4, 0.07, 160, 1, false);
    const disk = new THREE.Mesh(geo, [edgeMat, faceMat, edgeMat]);
    const group = new THREE.Group();
    group.add(disk);
    scene.add(group);

    // soft safelight pool under the wafer (radial gradient, fades to fully transparent)
    const gc = document.createElement("canvas");
    gc.width = gc.height = 256;
    const g2 = gc.getContext("2d");
    const grad = g2.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0, "rgba(255,197,61,0.55)");
    grad.addColorStop(0.45, "rgba(255,197,61,0.16)");
    grad.addColorStop(1, "rgba(255,197,61,0)");
    g2.fillStyle = grad;
    g2.fillRect(0, 0, 256, 256);
    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(5.6, 5.6),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(gc), transparent: true, depthWrite: false })
    );
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = -0.6;
    scene.add(glow);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xfff2cc, 1.2);
    key.position.set(3, 6, 4);
    scene.add(key);

    function resize() {
      const w = canvasEl.clientWidth, h = canvasEl.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(canvasEl);
    resize();

    let tx = 0, ty = 0;
    addEventListener("pointermove", (e) => { tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5; });

    let scanStart = -1;
    api.expose = () => { scanStart = reduceMotion ? -1 : performance.now(); };
    const apply = (c) => {
      tex.image = c;
      tex.needsUpdate = true;
      uniforms.uHasMap.value = 1;
      $("stageNote").hidden = true;
    };
    api.setImage = apply;
    if (pendingImage) apply(pendingImage);

    const t0 = performance.now();
    function frame(now) {
      const t = (now - t0) / 1000;
      uniforms.uTime.value = t;
      if (scanStart > 0) {
        const k = (now - scanStart) / 1100;
        uniforms.uScan.value = k > 1 ? -2 : 1.1 - k * 1.2;
        if (k > 1) scanStart = -1;
      }
      const spin = reduceMotion ? 0 : t * 0.12;
      group.rotation.y += (spin + tx * 0.6 - group.rotation.y) * 0.06;
      group.rotation.x += (0.12 + ty * 0.35 - group.rotation.x) * 0.06;
      group.position.y = reduceMotion ? 0 : Math.sin(t * 0.8) * 0.05;
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }).catch(() => {
    $("stageNote").textContent = "3D view unavailable; predictions still work.";
  });
  return api;
})();

// ---------------- Inputs ----------------

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("that file isn't an image the browser can read"));
    img.src = src;
  });
}

async function useFile(file) {
  if (!file) return;
  document.querySelectorAll(".example").forEach((b) => b.setAttribute("aria-pressed", "false"));
  try {
    const url = URL.createObjectURL(file);
    await classify(await loadImage(url), null);
    URL.revokeObjectURL(url);
  } catch (err) {
    $("status").textContent = `Couldn't open the image: ${err.message}.`;
  }
}

async function setupExamples() {
  const box = $("examples");
  let list = [];
  try {
    list = await (await fetch("examples/index.json")).json();
  } catch {
    box.innerHTML = '<p class="hint">Examples are unavailable.</p>';
    return;
  }
  for (const file of list) {
    const truth = file.split("__")[0];
    const b = document.createElement("button");
    b.className = "example";
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.dataset.truth = truth;
    b.innerHTML = `<img src="examples/${file}" alt="" loading="lazy"><span>${truth}</span>`;
    b.addEventListener("click", async () => {
      document.querySelectorAll(".example").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      await classify(await loadImage(`examples/${file}`), truth);
    });
    box.appendChild(b);
  }
}

const drop = $("drop");
$("file").addEventListener("change", (e) => useFile(e.target.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => useFile(e.dataTransfer.files[0]));

// ---------------- Model card ----------------

async function setupModelCard() {
  let m;
  try {
    m = await (await fetch("metrics.json")).json();
  } catch {
    $("stats").innerHTML = '<p class="hint">Metrics file not found.</p>';
    return;
  }
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  $("stats").innerHTML = [
    [pct(m.accuracy), "accuracy"],
    [m.macro_f1.toFixed(3), "macro-F1"],
    [m.n.toLocaleString(), "test images"],
  ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

  const names = { MULTI_CLASS: "Roboflow wafer maps", WM811K_IMG: "WM-811K 32×32 maps", IESA: "IESA challenge images" };
  $("sources").innerHTML = Object.entries(m.per_source_accuracy).map(([s, v]) =>
    `<div class="source"><span>${names[s] || s} <small>${v.n} images</small></span><span>${pct(v.accuracy)}</span>` +
    `<span class="track"><span class="fill" style="width:${v.accuracy * 100}%"></span></span></div>`).join("");

  const cm = m.confusion_matrix, labels = m.labels, n = labels.length;
  const grid = $("matrix");
  grid.style.gridTemplateColumns = `auto repeat(${n}, minmax(26px, 1fr))`;
  let html = "<div></div>" + labels.map((l) => `<div class="lab col">${l}</div>`).join("");
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="lab">${labels[i]}</div>`;
    row.forEach((v, j) => {
      const f = v / total;
      const color = i === j ? `rgba(255,197,61,${0.12 + f * 0.75})` : `rgba(255,122,217,${v ? 0.14 + f * 0.9 : 0.03})`;
      const ink = i === j && f > 0.45 ? "#2a1b00" : "var(--mist)";
      html += `<div class="cell" style="background:${color};color:${ink}" title="True ${labels[i]}, predicted ${labels[j]}: ${v}">${v || ""}</div>`;
    });
  });
  grid.innerHTML = html;
}

setupExamples();
setupModelCard();
