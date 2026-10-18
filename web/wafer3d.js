// Interactive wafer: a three.js disc whose rim carries nine arcs, one per defect class.
//
// The arcs are an index of the classes, not a defect locator: the model returns one label for the whole image.
// An arc's brightness follows the model's probability for its class (idle: a slow breathing wave). The selected
// image is mapped onto the wafer face. three.js is loaded lazily, with a 2D canvas fallback when WebGL is unavailable.

import { THREE_SRC, loadScript } from "./inference.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

const VERT = `
  varying vec2 vUv; varying vec3 vNormalW; varying vec3 vPosW;
  void main() {
    vUv = uv;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vPosW = w.xyz; vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

const FRAG = `
  uniform sampler2D uMap; uniform float uHasMap; uniform float uTime; uniform float uScan;
  uniform vec3 uCol[9]; uniform float uInt[9];
  varying vec2 vUv; varying vec3 vNormalW; varying vec3 vPosW;
  const float PI = 3.14159265;
  vec3 film(float t) { return 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + t)); }
  void main() {
    vec2 p = vUv * 2.0 - 1.0;
    float r = length(p);
    if (r > 1.0) discard;
    vec3 V = normalize(cameraPosition - vPosW);
    float cosT = clamp(dot(normalize(vNormalW), V), 0.0, 1.0);
    // thin-film interference: hue shifts with viewing angle and radius, like a polished wafer
    vec3 irid = film(1.6 * cosT + 0.55 * r + 0.04 * sin(uTime * 0.4));
    vec3 base = vec3(0.34, 0.36, 0.46);
    float img = texture2D(uMap, vec2(vUv.x, 1.0 - vUv.y)).r;
    vec3 surface = mix(base, vec3(0.06) + vec3(img) * 0.92, uHasMap);
    vec3 col = mix(surface, surface * (0.55 + 0.75 * irid), mix(0.5, 0.28, uHasMap));
    // die grid
    vec2 g = abs(fract(vUv * 26.0) - 0.5);
    col = mix(col, col * 0.55, smoothstep(0.47, 0.5, max(g.x, g.y)) * 0.55);
    col += pow(1.0 - cosT, 3.0) * 0.22;

    // nine class arcs around the rim
    float ang = atan(p.y, p.x);
    float t = (ang + PI) / (2.0 * PI) * 9.0;
    float idx = floor(t);
    float f = fract(t);
    float inSector = smoothstep(0.0, 0.06, f) * (1.0 - smoothstep(0.94, 1.0, f));
    float band = smoothstep(0.80, 0.845, r) * (1.0 - smoothstep(0.965, 0.99, r));
    vec3 ac = vec3(0.0); float ai = 0.0;
    for (int i = 0; i < 9; i++) { if (abs(idx - float(i)) < 0.5) { ac = uCol[i]; ai = uInt[i]; } }
    float m = band * inSector;
    col = mix(col, ac, m * (0.28 + 0.62 * ai));
    col += ac * m * ai * 0.55;
    // soft halo just outside the band
    col += ac * ai * 0.35 * smoothstep(0.985, 0.995, r) * (1.0 - smoothstep(0.995, 1.0, r)) * inSector;

    // exposure scan
    float d = abs(vUv.y - uScan);
    col += vec3(1.0, 0.77, 0.24) * smoothstep(0.08, 0.0, d) * 0.8;
    gl_FragColor = vec4(col, 1.0);
  }`;

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255);
}

/**
 * @param {object} o
 * @param {HTMLCanvasElement} o.canvas
 * @param {HTMLElement} o.stage        container observed for visibility
 * @param {{name:string,color:string}[]} o.classes
 * @param {(msg:string)=>void} o.onNote   status text (loading / unavailable); empty string clears it
 */
export function createWafer({ canvas, stage, classes, onNote }) {
  const n = classes.length;
  const target = new Float32Array(n).fill(0.35); // desired arc intensity per class
  const current = new Float32Array(n).fill(0.35);
  let hover = -1;
  let hasPrediction = false;
  let image = null;
  let impl = null; // {render(t, dt), resize(), setImage(canvas), scan()}
  let visible = true;
  let started = false;
  let scheduled = false;
  let last = 0;

  // rotation state (shared by the WebGL and 2D renderers)
  const rot = { y: 0.6, x: 0.34, vy: 0, dragging: false, lastX: 0, lastY: 0 };

  function motion() {
    return !reduceMotion.matches;
  }

  function updateTargets(t) {
    for (let i = 0; i < n; i++) {
      let v;
      if (hasPrediction) v = target[i];
      else v = 0.3 + (motion() ? 0.16 * Math.sin(t * 1.3 + i * 0.7) : 0.08);
      if (hover >= 0) v = i === hover ? Math.max(v, 0.95) : v * 0.4;
      current[i] += (v - current[i]) * (motion() ? 0.12 : 1);
    }
  }

  function frame(now) {
    scheduled = false;
    const t = now / 1000;
    const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016);
    last = now;
    if (!rot.dragging) {
      rot.y += rot.vy + (motion() ? 0.3 * dt : 0); // drag inertia plus a slow turntable spin
      rot.vy *= 0.94;
      if (Math.abs(rot.vy) < 0.0004) rot.vy = 0;
    }
    updateTargets(t);
    impl?.render(t, dt);
    if (visible && !document.hidden && motion()) request();
    else if (!settled()) request();
  }

  function settled() {
    for (let i = 0; i < n; i++) if (Math.abs(current[i] - target[i]) > 0.002 && hasPrediction) return false;
    return true;
  }

  function request() {
    if (scheduled || !started) return;
    scheduled = true;
    requestAnimationFrame(frame);
  }

  // ---- pointer + keyboard rotation ----
  canvas.addEventListener("pointerdown", (e) => {
    rot.dragging = true;
    rot.lastX = e.clientX;
    rot.lastY = e.clientY;
    rot.vy = 0;
    canvas.setPointerCapture(e.pointerId);
    request();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!rot.dragging) return;
    const dx = e.clientX - rot.lastX;
    const dy = e.clientY - rot.lastY;
    rot.lastX = e.clientX;
    rot.lastY = e.clientY;
    rot.y += dx * 0.012;
    rot.vy = dx * 0.004;
    rot.x = Math.max(-0.35, Math.min(1.05, rot.x + dy * 0.008));
    request();
  });
  const endDrag = () => {
    rot.dragging = false;
    request();
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", () => {
    rot.x = 0.34;
    rot.vy = 0;
    request();
  });
  canvas.addEventListener("keydown", (e) => {
    const step = 0.18;
    if (e.key === "ArrowLeft") rot.y -= step;
    else if (e.key === "ArrowRight") rot.y += step;
    else if (e.key === "ArrowUp") rot.x = Math.max(-0.35, rot.x - step);
    else if (e.key === "ArrowDown") rot.x = Math.min(1.05, rot.x + step);
    else return;
    e.preventDefault();
    request();
  });

  // ---- renderers ----
  function buildWebGL(THREE) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    camera.position.set(0, 6.6, 9.6);
    camera.lookAt(0, -0.2, 0);

    const blank = document.createElement("canvas");
    blank.width = blank.height = 8;
    const tex = new THREE.CanvasTexture(blank);
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    const colors = classes.map((c) => new THREE.Vector3(...hexToRgb(c.color)));
    const uniforms = {
      uMap: { value: tex }, uHasMap: { value: 0 }, uTime: { value: 0 }, uScan: { value: -2 },
      uCol: { value: colors }, uInt: { value: Array.from(current) },
    };
    const faceMat = new THREE.ShaderMaterial({ uniforms, vertexShader: VERT, fragmentShader: FRAG });
    const edgeMat = new THREE.MeshStandardMaterial({ color: 0x9a9ab8, metalness: 0.9, roughness: 0.28 });
    const disk = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, 0.08, 160, 1, false), [edgeMat, faceMat, edgeMat]);
    const group = new THREE.Group();
    group.add(disk);
    scene.add(group);

    // soft glow pool under the wafer
    const gc = document.createElement("canvas");
    gc.width = gc.height = 256;
    const g2 = gc.getContext("2d");
    const grad = g2.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0, "rgba(139,123,255,0.55)");
    grad.addColorStop(0.5, "rgba(61,224,255,0.14)");
    grad.addColorStop(1, "rgba(61,224,255,0)");
    g2.fillStyle = grad;
    g2.fillRect(0, 0, 256, 256);
    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(6.4, 6.4),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(gc), transparent: true, depthWrite: false })
    );
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = -0.7;
    scene.add(glow);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xfff2dd, 1.1);
    key.position.set(3, 6, 4);
    scene.add(key);

    let scanStart = -1;
    return {
      resize() {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.position.z = w / h < 0.9 ? 11.2 : 9.6;
        camera.updateProjectionMatrix();
      },
      setImage(c) {
        tex.image = c;
        tex.needsUpdate = true;
        uniforms.uHasMap.value = 1;
      },
      scan(now) {
        scanStart = now;
      },
      render(t) {
        uniforms.uTime.value = t;
        for (let i = 0; i < n; i++) uniforms.uInt.value[i] = current[i];
        if (scanStart >= 0) {
          const k = (performance.now() - scanStart) / 1100;
          uniforms.uScan.value = k > 1 ? -2 : 1.1 - k * 1.2;
          if (k > 1) scanStart = -1;
        }
        group.rotation.y = rot.y;
        group.rotation.x = rot.x - 0.34;
        group.position.y = motion() ? Math.sin(t * 0.8) * 0.05 : 0;
        renderer.render(scene, camera);
      },
      scanning: () => scanStart >= 0,
    };
  }

  function buildFallback() {
    const ctx = canvas.getContext("2d");
    const ring = (i) => [(-Math.PI + (i / n) * 2 * Math.PI) + 0.04, (-Math.PI + ((i + 1) / n) * 2 * Math.PI) - 0.04];
    return {
      resize() {
        const dpr = Math.min(devicePixelRatio, 2);
        canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
        canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
      },
      setImage() {},
      scan() {},
      render() {
        const w = canvas.width, h = canvas.height;
        const R = Math.min(w, h) * 0.34;
        const sy = 0.55 + 0.45 * Math.cos(rot.x); // tilt squashes the disc vertically
        ctx.clearRect(0, 0, w, h);
        ctx.save();
        ctx.translate(w / 2, h / 2);
        ctx.scale(1, sy);
        ctx.rotate(rot.y);
        ctx.beginPath();
        ctx.arc(0, 0, R, 0, Math.PI * 2);
        ctx.fillStyle = "#8c8fa8";
        ctx.fill();
        if (image) {
          ctx.save();
          ctx.clip();
          ctx.drawImage(image, -R, -R, 2 * R, 2 * R);
          ctx.restore();
        }
        ctx.lineCap = "round";
        for (let i = 0; i < n; i++) {
          const [a0, a1] = ring(i);
          ctx.beginPath();
          ctx.arc(0, 0, R * 0.92, a0, a1);
          ctx.strokeStyle = classes[i].color;
          ctx.globalAlpha = 0.25 + 0.75 * current[i];
          ctx.lineWidth = R * 0.1;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.restore();
      },
      scanning: () => false,
    };
  }

  async function start() {
    if (started) return;
    started = true;
    onNote("Preparing the 3D view…");
    try {
      await loadScript(THREE_SRC);
      impl = buildWebGL(window.THREE);
    } catch (err) {
      console.warn("3D view unavailable, using the 2D fallback:", err);
      try {
        impl = buildFallback();
      } catch {
        onNote("3D view unavailable. Predictions still work.");
        return;
      }
    }
    onNote("");
    new ResizeObserver(() => {
      impl.resize();
      request();
    }).observe(canvas);
    impl.resize();
    if (image) impl.setImage(image);
    request();
  }

  // lazy: load three.js when the stage is about to be visible
  const io = new IntersectionObserver(
    (entries) => {
      visible = entries.some((e) => e.isIntersecting);
      if (visible) {
        start();
        request();
      }
    },
    { rootMargin: "200px" }
  );
  io.observe(stage);
  reduceMotion.addEventListener?.("change", request);
  document.addEventListener("visibilitychange", request);

  return {
    /** Wear a 224x224 canvas on the wafer face. */
    setImage(c) {
      image = c;
      impl?.setImage(c);
      request();
    },
    /** probs: number[] in class order, or null to return to the idle state. */
    setProbabilities(probs) {
      hasPrediction = !!probs;
      for (let i = 0; i < n; i++) target[i] = probs ? 0.1 + 0.9 * probs[i] : 0.35;
      request();
    },
    highlight(i) {
      hover = i;
      request();
    },
    /** Play the exposure scan (3D renderer only). */
    expose() {
      if (motion()) impl?.scan?.(performance.now());
      request();
    },
  };
}
