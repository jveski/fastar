# Demos and example source

These are the live demos **and** the examples. Each HTML page imports the library
directly; its tracking and rendering code is unbundled and readable. Assets and
small feature-specific helpers live beside the pages. The build copies this
directory unchanged, so the source linked from a demo is the code that runs it.

## Run locally

```sh
npm run build
npx serve dist/site
```

Open `http://localhost:3000/demo/demos.html`. Phones need HTTPS for camera,
microphone and motion access. Use an HTTPS host or run
`npx cloudflared tunnel --url http://localhost:3000` and open the tunnel's
`/demo/demos.html` URL. Serve **dist/site**, not the repository root.

| Demo / source | What to try |
| --- | --- |
| [Image](index.html) | Show `target.png` on another screen or print it 20 cm wide. `?targets=<url>` accepts a compiled target file. |
| [Video](video.html) | Open [the player](video/screen.html) on a second screen. The cube follows recovered video time. [Overlay](video/overlay.js). |
| [Paper](paper.html) | Print `paper/target.png` 25 cm wide, find it flat, then curl it. [Liquid simulation](paper/fluid.js) uses `surface.sample()` and gravity. |
| [Several targets](multi.html) | Open [four clips](multi/screen.html) on another screen and view them from about 30 cm. Inspect lazy content states and change target priority. [Overlay](multi/overlay.js). |
| [Timecode](timecode.html) | [Play](timecode/screen.html) on speakers that reach 20 kHz; listen on another device, optionally with image tracking. `?key=my-set` is passed to the player. |
| [Performance](performance.html) | Add artificial frame-handling load, watch real performance hints reduce resolution and particles, then remove the load to recover. Restart with synchronized display off to compare. |

## Generate targets again

The committed targets are ready to use. To regenerate the procedural artwork:

```sh
npx tsx scripts/example-target.ts
npx tsx scripts/example-video.ts   # needs ffmpeg
npx tsx scripts/example-paper.ts
npx tsx scripts/example-multi.ts   # needs ffmpeg
npm run build
```

Physical widths are in metres: image 0.20, paper 0.25, video 0.30 and each of
the four clips 0.14. Recompile for a different display size or your own artwork:

```sh
npx fastar-compile -o site/demo/targets.bin my-poster.jpg:0.3
npx fastar-compile --points 120 -o site/demo/video/targets.bin site/demo/video/clip.mp4:0.6
npx fastar-compile --points 60 -o site/demo/multi/targets.bin site/demo/multi/clips/{amber,cobalt,jade,coral}.mp4:0.2
```

Or use [the browser compiler](../compile.html), which links directly to the image
demo with your compiled targets.

The timecode player generates audio in the browser without a checked-in recording.
For a longer set or prerecorded music, use the CLI with the same key as the listener:

```sh
node dist/fastar-timecode.js generate -o timecode.wav --duration 1h --key my-set
node dist/fastar-timecode.js mix -o out.wav set.wav --key my-set
node dist/fastar-timecode.js decode recording.wav
```

Use a lossless microphone recording for decoding: compressed audio often drops
the 19–20 kHz band. Keep playback volume low; inaudibility is not a reason to raise it.
