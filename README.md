# fastar

Fast image-target AR for the mobile web.
Works in iOS Safari and Android Chrome without WebXR.

**[Demos](https://jveski.github.io/fastar/demo/demos.html) · [Example source](site/demo) · [Target compiler](https://jveski.github.io/fastar/compile.html)**

Each demo runs in the browser and links to its readable source: the demos are the examples.

Shipped as a single ES module with no dependencies. Import it as below, or download [`fastar.js`](https://github.com/jveski/fastar/releases/latest/download/fastar.js) from the latest release.

```html
<video id="video"></video>
<button id="start">Start</button>

<script type="module">
  import { ImageTracker } from 'https://jveski.github.io/fastar/fastar.js';

  const ar = await ImageTracker.create({ targets: './targets.bin' });
  ar.onFrame(({ poses }) => {
    // poses.get(0)?.matrix: model-view matrix; ar.projectionMatrix: matching projection
  });
  document.getElementById('start').onclick = () => ar.start({ video: document.getElementById('video') });
</script>
```

## Video targets

Short looping videos work as targets too, and poses report where the video is in its playback. Nothing in the video needs to hold still: tracking works even when the whole picture keeps changing. [Demo](https://jveski.github.io/fastar/demo/video.html) · [source](site/demo/video.html)

```sh
npx fastar-compile -o targets.bin loop.mp4:1.2  # needs ffmpeg; or compileVideo() in the browser
```

```js
ar.onFrame(({ poses }) => {
  const p = poses.get(0);
  if (p?.videoTimeLocked) console.log(p.videoTime); // seconds into the clip
});
```

## Bending targets

A printed page doesn't stay flat. Turn on `deformable` and each pose also carries the page's shape as it curls, flexes or lifts a corner. Any image target works; no special compilation is needed. [Demo](https://jveski.github.io/fastar/demo/paper.html) · [source](site/demo/paper.html)

```js
const ar = await ImageTracker.create({ targets: './targets.bin', deformable: true });
ar.onFrame(({ poses }) => {
  const p = poses.get(0);
  const s = p?.surface?.sample(0.5, 0.5); // middle of the page, in the coordinates of p.matrix
  // s.position, s.normal, s.tangentU, s.tangentV; p.gravity: down, where motion sensors are available
});
```

`surface.positions` holds the mesh itself (`cols` x `rows` vertices from the top-left corner), and `surface.fillGrid()` fills render geometry at any resolution. With three.js, `three.surface(0, { material })` adds a mesh that follows the page, textured like the target image. `deformable: { targets, grid, smoothing }` chooses which targets bend, the mesh resolution (vertices along the long side, default 9) and how much small shape changes are eased.

Find the page while it is roughly flat, then bend it. Smooth bends work; creases and folds don't, and video targets are always rigid. The page needs detail all over: every part of it must be tracked to be shaped.

## Several targets

Compile several images or videos into one file and every one of them is tracked, all at once. `poses` holds each target in view by its index, in the order they were compiled; `ar.targetIndex(name)` looks one up by name. [Demo](https://jveski.github.io/fastar/demo/multi.html) · [source](site/demo/multi.html) (four videos on one screen, with content loaded on demand)

```sh
npx fastar-compile -o targets.bin poster.png:0.42 cover.jpg:0.21 loop.mp4:1.2
```

Each target's content can be loaded only when it is needed: when the target is found, or nearly found as it comes into view (the `near` event). Content out of view for a while is unloaded to make room. How much stays loaded follows the device (`budget: 'auto'`): more on fast phones and computers, less on slow ones, and less still while the device cannot keep up.

```js
const ar = await ImageTracker.create({ targets: './targets.bin' });
const three = new ThreeAdapter(ar, { THREE, scene, camera, content: { budget: 'auto', keepAlive: 5000 } });
ar.targets.forEach((t, i) => three.content(i, async ({ signal, name }) => {
  const gltf = await loader.loadAsync(`./models/${name}.glb`); // added to target i's anchor once loaded
  return gltf.scene;                                           // removed and disposed once unloaded
}));
```

Without three.js, `ar.content({ load, unload, show, hide })` does the same for anything, and in A-Frame `fastar-content="src: model.glb"` (or a video) on a `fastar-target` entity loads it only once that target is found.

The time spent tracking per camera frame is bounded (`budget: { trackMs }`, by default 60% of the frame interval), so a slow phone does not fall behind as targets come into view. Every target in view is still tracked: the most prominent ones (largest, wholly in view, nearest the centre) keep full resolution, and the others are tracked with fewer points, or on fewer frames, when time runs short. `ar.setPriority(index, weight)` favours a target, and `ar.stats.targets` shows each target's priority, points and update rate. `maxTrack` limits how many targets are tracked at once.

## Audio timecode

Share one timeline: a timecode encoded as sound, too high to hear (sweeps between 19 and 20 kHz). Turn on `timecode` and every frame carries the set time it was captured at. [Demo](https://jveski.github.io/fastar/demo/timecode.html) · [source](site/demo/timecode.html)

```js
await ar.start({ video, timecode: { key: 'my-set' } });
ar.onFrame(({ timecode }) => {
  if (timecode?.locked) console.log(timecode.time);
});
```

Make the timecode with `fastar-timecode`, using the same key:

```sh
npx fastar-timecode generate -o timecode.wav --duration 4h --key my-set
npx fastar-timecode mix -o out.wav set.wav --key my-set
```

Without image tracking, `TimecodeListener` listens on its own:

```js
start.onclick = async () => {
  const tc = new TimecodeListener({ key: 'my-set' }); // in the tap: iOS only starts audio from a gesture
  await tc.start();
  tc.on('lock', () => console.log(tc.time)); // tc.timeAt(performance.now() ms) for any moment
};
```

## Slow devices

Tracking adjusts itself to the phone it runs on. When it cannot keep up with every camera frame, it tracks every 2nd one, evenly, rather than whichever frames it happens to reach, and looking for targets not yet found takes less of the time. [Demo](https://jveski.github.io/fastar/demo/performance.html) · [source](site/demo/performance.html)

With `syncDisplay: true` (draw `FrameEvent.image` as the background, as the demos do) every camera frame is shown, at the camera's rate, with the pose measured on it. Frames are shown a steady delay after capture so that most have been tracked by then. The delay follows how long tracking takes on the device, is never more than `syncDisplay: { maxDelay }` (80 ms by default), and is zero while no target is in view.

Rendering is up to the app, so the tracker gives hints. `ar.performance` says how hard the device is pressed (`level`, 0 to 3), what holds it back (`bottleneck`: `'render'`, `'tracking'` or `'none'`), and suggests a `pixelRatio` and a level of `effects` to render with. Apply them when they change:

```js
ar.on('performance', (p) => {
  renderer.setPixelRatio(Math.min(p.pixelRatio, 2));
  shadows.enabled = p.effects === 'full';
});
```

In A-Frame the scene emits `fastar-performance` with the same details. `ar.performance` also holds the measurements behind them: the camera's frame interval, the share of the time spent handling frames on the page and looking for targets, late display frames, and the display delay.
