// In-browser inference with ONNX Runtime Web (WebAssembly, CPU).
//
// Contract with the Python package (edgedefectnet/preprocessing.py):
//   grayscale L = round(0.299 R + 0.587 G + 0.114 B), resized to 224x224, then (L / 255 - 0.45) / 0.23.
// The model's 9 logits are in the order of classes.json (alphabetical, = CLASS_NAMES in Python).

export const SIZE = 224;
const MEAN = 0.45;
const STD = 0.23;

// Pinned versions. The browser refuses a script whose bytes do not match the SRI hash, and each file is
// served identically by both CDNs, so the second origin is a true fallback. ONNX Runtime fetches its .wasm
// binary at run time from the same pinned directory; that file cannot carry an integrity attribute.
export const ORT_SRC = {
  urls: [
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.js",
    "https://unpkg.com/onnxruntime-web@1.20.1/dist/ort.wasm.min.js",
  ],
  integrity: "sha384-6hgW9I+eQKl5CrHLstudPd4L3E2eTYHm+qNKvf4pI+RZnfBJILOHAiaalHV+trgG",
};
export const THREE_SRC = {
  urls: [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
  ],
  integrity: "sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu",
};

/** Load a classic script from the first CDN that works, verifying its SRI hash. Resolves with the URL used. */
export function loadScript({ urls, integrity }) {
  return new Promise((resolve, reject) => {
    const tryNext = (i) => {
      if (i >= urls.length) return reject(new Error(`could not load ${urls[0]}`));
      const el = document.createElement("script");
      el.src = urls[i];
      el.integrity = integrity;
      el.crossOrigin = "anonymous";
      el.onload = () => resolve(urls[i]);
      el.onerror = () => {
        el.remove();
        tryNext(i + 1);
      };
      document.head.appendChild(el);
    };
    tryNext(0);
  });
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model download failed (HTTP ${resp.status})`);
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || !total) {
    onProgress(null, 0);
    return new Uint8Array(await resp.arrayBuffer());
  }
  const reader = resp.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total, got);
  }
  return out;
}

let runtimePromise = null;
let sessionPromise = null;

/** Start loading the small ONNX Runtime script early (no model bytes yet). */
export function warmRuntime() {
  if (!runtimePromise) {
    runtimePromise = loadScript(ORT_SRC).then((url) => {
      const ort = window.ort;
      ort.env.wasm.wasmPaths = url.replace(/ort\.wasm\.min\.js$/, "");
      ort.env.wasm.numThreads = 1;
      return ort;
    });
    runtimePromise.catch(() => (runtimePromise = null));
  }
  return runtimePromise;
}

/** Resolve with {ort, session}; the model is downloaded once and reused. onStage receives status updates. */
export function getSession(onStage = () => {}) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      onStage({ stage: "runtime" });
      const ort = await warmRuntime();
      onStage({ stage: "download", fraction: 0, bytes: 0 });
      const bytes = await fetchWithProgress("model.onnx", (fraction, got) => onStage({ stage: "download", fraction, bytes: got }));
      onStage({ stage: "start" });
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      return { ort, session };
    })();
    sessionPromise.catch(() => (sessionPromise = null)); // allow a retry after a failure
  }
  return sessionPromise;
}

/** Draw an image into a 224x224 canvas (the size the model expects). */
export function toModelCanvas(img) {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  return c;
}

export function toTensorData(canvas) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, SIZE, SIZE);
  const out = new Float32Array(SIZE * SIZE);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const l = Math.round((data[p] * 299 + data[p + 1] * 587 + data[p + 2] * 114) / 1000); // PIL "L"
    out[i] = (l / 255 - MEAN) / STD;
  }
  return out;
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Run the model on a 224x224 canvas. Returns {probs, ms}. */
export async function predict(canvas, onStage) {
  const { ort, session } = await getSession(onStage);
  const t0 = performance.now();
  const input = new ort.Tensor("float32", toTensorData(canvas), [1, 1, SIZE, SIZE]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const probs = softmax(Array.from(out[session.outputNames[0]].data));
  return { probs, ms: performance.now() - t0 };
}
