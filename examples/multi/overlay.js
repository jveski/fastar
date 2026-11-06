// The AR content for one clip, loaded only once its clip is found (or
// nearly found): a shape standing on the clip that turns with the swirl at
// the clip's centre, grows and shrinks as it spreads and gathers, and
// shifts hue as the clip's colours turn. It follows the video time fastar
// recovers from the camera; until that time is known, it rests in its plain
// colour.
//
// Target units are metres, origin at the clip's centre, x right, y up, z out
// of the screen. `info` is the clip's JSON: its swirl (waves [amp, cycles,
// phase]: value(t) = sum of amp * sin(2 pi cycles t / loop + phase)), and the
// shape and colour of its overlay.

const geometries = {
  cube: (THREE, s) => new THREE.BoxGeometry(s, s, s),
  torusKnot: (THREE, s) => new THREE.TorusKnotGeometry(s * 0.32, s * 0.1, 96, 12),
  icosahedron: (THREE, s) => new THREE.IcosahedronGeometry(s * 0.55),
  cone: (THREE, s) => new THREE.ConeGeometry(s * 0.45, s, 32).rotateX(Math.PI / 2),
};

export function createOverlay(THREE, target, info) {
  const size = target.physicalWidth / 3;
  const base = new THREE.Color(info.color);
  const material = new THREE.MeshStandardMaterial({ color: base, roughness: 0.35, metalness: 0.1 });
  const mesh = new THREE.Mesh((geometries[info.shape] ?? geometries.cube)(THREE, size), material);
  mesh.position.z = size / 2;
  // Turning and scaling this group keeps the shape standing on the screen.
  const root = new THREE.Group();
  root.add(mesh);

  const waves = (list, t) => list.reduce((s, [a, n, ph]) => s + a * Math.sin((2 * Math.PI * n * t) / info.loop + ph), 0);
  // Chroma of the base colour (BT.601 YCbCr, as the clip's colours are made), turned like the clip's colours.
  const srgb = base.getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace);
  const Y = 0.299 * srgb.r + 0.587 * srgb.g + 0.114 * srgb.b;
  const cb = (srgb.b - Y) / 1.772, cr = (srgb.r - Y) / 1.402;
  const clamp = (x) => Math.min(1, Math.max(0, x));
  const tinted = new THREE.Color();

  /** Video time in seconds, or null while it is not known. */
  root.userData.update = (t) => {
    if (t === null || t === undefined) {
      root.rotation.z = 0;
      root.scale.setScalar(1);
      material.color.copy(base);
      material.emissive.setRGB(0, 0, 0);
      return;
    }
    root.rotation.z = waves(info.spin, t);
    root.scale.setScalar(Math.exp(-waves(info.zoom, t)));
    const h = (2 * Math.PI * info.hueCycles * t) / info.loop;
    const u = cb * Math.cos(h) - cr * Math.sin(h), v = cb * Math.sin(h) + cr * Math.cos(h);
    tinted.setRGB(clamp(Y + 1.402 * v), clamp(Y - 0.344136 * u - 0.714136 * v), clamp(Y + 1.772 * u), THREE.SRGBColorSpace);
    material.color.copy(tinted);
    material.emissive.copy(tinted).multiplyScalar(0.35);
  };
  return root;
}
