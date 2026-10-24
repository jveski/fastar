// AR overlay for the marbled-ink clip: the cube from the image demo, kept
// in step with the video. It turns with the swirl at the centre of the
// screen, grows and shrinks as that swirl spreads and gathers, and shifts
// hue as the clip's colours turn. All of it comes from the video time fastar
// recovers from the camera, so the cube only moves with the picture while
// that time is right; until it is known, the cube rests in its plain colour.
//
// Target units are metres, origin at the screen centre, x right, y up, z out
// of the screen. scene.json describes the clip: for a list of waves
// [amp, cycles, phase], value(t) = sum of amp * sin(2 pi cycles t / loop + phase).

const BASE = 0x2d7ff9;

export function createOverlay(THREE, target, info) {
  const size = target.physicalWidth / 3;
  const material = new THREE.MeshStandardMaterial({ color: BASE });
  const cube = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), material);
  cube.position.z = size / 2;
  // Turning and scaling this group keeps the cube standing on the screen.
  const root = new THREE.Group();
  root.add(cube);

  const waves = (list, t) => list.reduce((s, [a, n, ph]) => s + a * Math.sin((2 * Math.PI * n * t) / info.loop + ph), 0);
  const base = new THREE.Color(BASE);
  // Chroma of the base colour (BT.601 YCbCr, as the clip's colours are
  // made), turned like the clip's colours.
  const srgb = base.getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace);
  const Y = 0.299 * srgb.r + 0.587 * srgb.g + 0.114 * srgb.b;
  const cb = (srgb.b - Y) / 1.772, cr = (srgb.r - Y) / 1.402;
  const clamp = (x) => Math.min(1, Math.max(0, x));
  const tinted = new THREE.Color();

  return {
    root,
    /** Video time in seconds, or null while it is not known. */
    update(t) {
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
    },
  };
}
