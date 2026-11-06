# fastar

Fast image-target AR for the mobile web.
Works in iOS Safari and Android Chrome without WebXR.

**[Live demo](https://jveski.github.io/fastar/demo/) · [Target compiler](https://jveski.github.io/fastar/compile.html)**

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

Short looping videos work as targets too, and poses report where the video is in its playback. Nothing in the video needs to hold still: tracking works even when the whole picture keeps changing. [Demo](https://jveski.github.io/fastar/demo/video.html) · [example](examples/video)

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

A printed page doesn't stay flat. Turn on `deformable` and each pose also carries the page's shape as it curls, flexes or lifts a corner. Any image target works; no special compilation is needed. [Demo](https://jveski.github.io/fastar/demo/paper.html) · [example](examples/paper)

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
