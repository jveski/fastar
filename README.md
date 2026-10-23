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

