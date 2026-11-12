// Liquid on a bending page. Drops live on the paper's surface: gravity pulls
// them along it (only the part of gravity along the surface where they are),
// so they run downhill, collect in the trough of a curled page and race down
// a tilted one. A light push between neighbours makes them pool and spread
// like a liquid; damping stands in for the paper's drag. Drops that run off
// an edge, or hang under a page bent past vertical, fall off and fade. A
// spring at the page's highest point keeps it flowing.
//
// Everything is in the target's coordinates (metres, origin at the page
// centre, x right, y up, z out of the print), so the drops are drawn in the
// target's anchor and follow the page as it moves and bends.

const STEP = 1 / 120;

export function createFluid(THREE, target, o = {}) {
  const W = target.physicalWidth, H = (W * target.height) / target.width;
  const MAX = o.count ?? 1500;
  const R = o.radius ?? 0.0022; // drop radius (m)
  const G = o.gravity ?? 3; // m/s^2 along the page: slower than real, easier to watch
  const DAMP = o.damping ?? 2.2; // 1/s
  const SPACING = 2.4 * R; // drops push apart closer than this
  const PUSH = 1.2; // m/s^2 at contact
  const RATE = o.rate ?? 160; // new drops per second from the spring
  const LIFE = 1.2; // seconds a falling drop lasts

  // State: on the page at (u, v) metres from the bottom-left corner with velocity (vu, vv),
  // or falling at p with velocity q (target axes), for `life` more seconds.
  const u = new Float32Array(MAX), v = new Float32Array(MAX), vu = new Float32Array(MAX), vv = new Float32Array(MAX);
  const p = new Float32Array(3 * MAX), q = new Float32Array(3 * MAX), life = new Float32Array(MAX);
  const falling = new Uint8Array(MAX);
  let n = 0;

  // Neighbour grid over the page, one cell per SPACING.
  const gw = Math.ceil(W / SPACING) + 1, gh = Math.ceil(H / SPACING) + 1;
  const start = new Int32Array(gw * gh + 1), items = new Int32Array(MAX), cellOf = new Int32Array(MAX);

  const geometry = new THREE.SphereGeometry(R, 10, 8);
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.15, metalness: 0.05, transparent: true, opacity: 0.92 });
  const drops = new THREE.InstancedMesh(geometry, material, MAX);
  drops.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  drops.frustumCulled = false;
  drops.count = 0;
  const root = new THREE.Group();
  root.add(drops);

  // The page itself: faint shading plus contour lines of height against gravity, so its shape shows.
  const surfaceMaterial = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: { up: { value: new THREE.Vector3(0, 1, 0) }, spacing: { value: 0.004 }, opacity: { value: 1 } },
    vertexShader: /* glsl */ `
      varying vec3 vPos;
      varying vec3 vNormal;
      void main() {
        vPos = position;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 up;
      uniform float spacing;
      uniform float opacity;
      varying vec3 vPos;
      varying vec3 vNormal;
      void main() {
        float h = dot(vPos, up) / spacing;
        float line = 1.0 - min(abs(fract(h - 0.5) - 0.5) / max(fwidth(h), 1e-4), 1.0);
        float shade = 0.5 + 0.5 * dot(normalize(vNormal), normalize(vec3(0.4, 0.7, 0.6)));
        vec3 tint = mix(vec3(0.05, 0.25, 0.55), vec3(0.6, 0.85, 1.0), shade);
        gl_FragColor = vec4(mix(tint, vec3(1.0), line), (0.18 * (1.0 - shade) + 0.55 * line) * opacity);
      }`,
  });

  const pt = { position: new Float32Array(3), normal: new Float32Array(3), tangentU: new Float32Array(3), tangentV: new Float32Array(3) };
  const flatSample = (uu, vv2, out) => {
    out.position[0] = uu - W / 2; out.position[1] = vv2 - H / 2; out.position[2] = 0;
    out.normal.set([0, 0, 1]); out.tangentU.set([1, 0, 0]); out.tangentV.set([0, 1, 0]);
    return out;
  };
  const sample = (s, uu, vv2) => (s ? s.sample(uu / W, vv2 / H, pt) : flatSample(uu, vv2, pt));

  const spring = { u: W / 2, v: H - 0.015 };
  let springAge = Infinity, debt = 0, acc = 0;

  /** The page's highest point (against gravity), a little inside its edges; nearest the middle among near-ties. */
  const heights = new Float32Array(11 * 9);
  function findSpring(s, down) {
    let top = -Infinity;
    for (let j = 0; j <= 8; j++) {
      for (let i = 0; i <= 10; i++) {
        const P = sample(s, 0.01 + ((W - 0.02) * i) / 10, 0.01 + ((H - 0.02) * j) / 8).position;
        const h = (heights[j * 11 + i] = -(P[0] * down[0] + P[1] * down[1] + P[2] * down[2]));
        if (h > top) top = h;
      }
    }
    let best = Infinity;
    for (let j = 0; j <= 8; j++) {
      for (let i = 0; i <= 10; i++) {
        if (heights[j * 11 + i] < top - 0.003) continue;
        const d = Math.hypot(i / 10 - 0.5, j / 8 - 0.5);
        if (d < best) { best = d; spring.u = 0.01 + ((W - 0.02) * i) / 10; spring.v = 0.01 + ((H - 0.02) * j) / 8; }
      }
    }
  }

  function add(uu, v2, jitter) {
    if (n >= MAX) return;
    const a = Math.random() * 2 * Math.PI, r = jitter * Math.sqrt(Math.random());
    u[n] = Math.min(W, Math.max(0, uu + r * Math.cos(a)));
    v[n] = Math.min(H, Math.max(0, v2 + r * Math.sin(a)));
    vu[n] = vv[n] = 0;
    falling[n] = 0;
    n++;
  }

  function remove(i) {
    n--;
    u[i] = u[n]; v[i] = v[n]; vu[i] = vu[n]; vv[i] = vv[n]; falling[i] = falling[n]; life[i] = life[n];
    for (let k = 0; k < 3; k++) { p[3 * i + k] = p[3 * n + k]; q[3 * i + k] = q[3 * n + k]; }
  }

  function fall(i, P) {
    falling[i] = 1;
    life[i] = LIFE;
    for (let k = 0; k < 3; k++) {
      p[3 * i + k] = P.position[k] + R * P.normal[k];
      q[3 * i + k] = vu[i] * P.tangentU[k] + vv[i] * P.tangentV[k];
    }
  }

  function step(s, down, dt) {
    // Neighbour grid of the drops on the page.
    start.fill(0);
    for (let i = 0; i < n; i++) {
      if (falling[i]) { cellOf[i] = -1; continue; }
      const c = Math.min(gh - 1, Math.max(0, (v[i] / SPACING) | 0)) * gw + Math.min(gw - 1, Math.max(0, (u[i] / SPACING) | 0));
      cellOf[i] = c;
      start[c + 1]++;
    }
    for (let c = 0; c < gw * gh; c++) start[c + 1] += start[c];
    const fill = start.slice(0, gw * gh);
    for (let i = 0; i < n; i++) if (cellOf[i] >= 0) items[fill[cellOf[i]]++] = i;

    const damp = Math.exp(-DAMP * dt);
    for (let i = n - 1; i >= 0; i--) {
      if (falling[i]) {
        for (let k = 0; k < 3; k++) {
          q[3 * i + k] += 2 * G * down[k] * dt;
          p[3 * i + k] += q[3 * i + k] * dt;
        }
        if ((life[i] -= dt) <= 0) remove(i);
        continue;
      }
      const P = sample(s, u[i], v[i]);
      const nd = P.normal[0] * down[0] + P.normal[1] * down[1] + P.normal[2] * down[2];
      // Hanging under the page: let go.
      if (nd > 0.35) { fall(i, P); continue; }
      let au = G * (down[0] * P.tangentU[0] + down[1] * P.tangentU[1] + down[2] * P.tangentU[2]);
      let av = G * (down[0] * P.tangentV[0] + down[1] * P.tangentV[1] + down[2] * P.tangentV[2]);
      // Push from close neighbours.
      const cx = Math.min(gw - 1, Math.max(0, (u[i] / SPACING) | 0)), cy = Math.min(gh - 1, Math.max(0, (v[i] / SPACING) | 0));
      for (let y = Math.max(0, cy - 1); y <= Math.min(gh - 1, cy + 1); y++) {
        for (let x = Math.max(0, cx - 1); x <= Math.min(gw - 1, cx + 1); x++) {
          const c = y * gw + x;
          for (let k = start[c]; k < start[c + 1]; k++) {
            const j = items[k];
            if (j === i) continue;
            const dx = u[i] - u[j], dy = v[i] - v[j], d2 = dx * dx + dy * dy;
            if (d2 >= SPACING * SPACING) continue;
            const d = Math.sqrt(d2) || 1e-6, f = PUSH * (1 - d / SPACING) / d;
            au += f * (dx || 1e-4 * (i - j));
            av += f * dy;
          }
        }
      }
      vu[i] = (vu[i] + au * dt) * damp;
      vv[i] = (vv[i] + av * dt) * damp;
      u[i] += vu[i] * dt;
      v[i] += vv[i] * dt;
      if (u[i] < 0 || u[i] > W || v[i] < 0 || v[i] > H) fall(i, sample(s, Math.min(W, Math.max(0, u[i])), Math.min(H, Math.max(0, v[i]))));
    }
  }

  const m = new THREE.Matrix4(), color = new THREE.Color(), slow = new THREE.Color(0x1f6fff), fast = new THREE.Color(0xc8f4ff);

  function draw(s) {
    for (let i = 0; i < n; i++) {
      let x, y, z, speed, fade = 1;
      if (falling[i]) {
        x = p[3 * i]; y = p[3 * i + 1]; z = p[3 * i + 2];
        speed = Math.hypot(q[3 * i], q[3 * i + 1], q[3 * i + 2]);
        fade = life[i] / LIFE;
      } else {
        const P = sample(s, u[i], v[i]);
        x = P.position[0] + R * P.normal[0]; y = P.position[1] + R * P.normal[1]; z = P.position[2] + R * P.normal[2];
        speed = Math.hypot(vu[i], vv[i]);
      }
      m.makeScale(fade, fade, fade).setPosition(x, y, z);
      drops.setMatrixAt(i, m);
      drops.setColorAt(i, color.copy(slow).lerp(fast, Math.min(1, speed / 0.25)));
    }
    drops.count = n;
    drops.instanceMatrix.needsUpdate = true;
    if (drops.instanceColor) drops.instanceColor.needsUpdate = true;
  }

  return {
    root,
    surfaceMaterial,
    get count() { return n; },
    /**
     * Advance by dt seconds on surface `s` (a fastar Surface, or null for a
     * flat page) with `down` the unit gravity direction in target axes.
     */
    update(s, down, dt) {
      surfaceMaterial.uniforms.up.value.set(-down[0], -down[1], -down[2]);
      if ((springAge += dt) > 0.25) { findSpring(s, down); springAge = 0; }
      debt += RATE * dt;
      for (; debt >= 1; debt--) add(spring.u, spring.v, 0.006);
      acc = Math.min(acc + dt, 4 * STEP);
      for (; acc >= STEP; acc -= STEP) step(s, down, STEP);
      draw(s);
    },
    /** Pour `count` drops at texture coordinates (tu, tv) of the page. */
    pour(tu, tv, count = 80) {
      for (let k = 0; k < count; k++) add(tu * W, tv * H, 0.012);
    },
    clear() { n = 0; drops.count = 0; },
  };
}
