# Minimal example

A blue cube on a printed image, rendered with three.js.

```sh
npm run build                      # builds dist/fastar.js
npx tsx scripts/example-target.ts  # (re)generates target.png and targets.bin here
npx serve .                        # serve the repository root
```

Open `http://localhost:3000/examples/basic/` and point the camera at `target.png`, either printed 20 cm wide or shown on another screen. Phones need HTTPS for camera and motion access: run `npx cloudflared tunnel --url http://localhost:3000` and open the printed `https://….trycloudflare.com/examples/basic/` URL, or use any HTTPS host. Serve the repository root so the page can import `../../dist/fastar.js`.

To use your own image, compile it with its printed width in metres:

```sh
npx tsx cli/compile.ts -o examples/basic/targets.bin my-poster.jpg:0.3
```
