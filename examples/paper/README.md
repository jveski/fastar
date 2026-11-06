# Paper example

Liquid on a printed page that bends: drops run downhill along the paper, pool where it curls and pour off its edges. The page is tracked as a deforming surface (`deformable: true`), and each pose carries its shape (`pose.surface`) and gravity (`pose.gravity`).

```sh
npm run build                     # builds dist/fastar.js
npx tsx scripts/example-paper.ts  # (re)generates target.png and targets.bin here
npx serve .                       # serve the repository root
```

Print `target.png` 25 cm wide (landscape A4 or US Letter), open `http://localhost:3000/examples/paper/` and point the camera at the page while it lies flat. Once it's found, pick it up and curl it. Phones need HTTPS for camera and motion access; see [the minimal example](../basic/README.md) for a tunnel.

`fluid.js` is the simulation: drops on the page in (u, v) coordinates, pulled by the part of gravity along the surface (`surface.sample(u, v)` gives position, normal and tangents), pushing apart from their neighbours. Bending targets need detail all over the page; `scripts/example-paper.ts` shows one way to make it.
