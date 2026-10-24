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
