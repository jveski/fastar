# Several video targets

Four different looping clips play side by side on one screen, and each is its
own video target. fastar tracks all of them at once, recovers where each clip
is in its playback, and puts a different shape on each that moves with its
clip.

Each clip's overlay is loaded only when that clip is found (or nearly found,
as it comes into view). The page fetches the clip's description
(`clips/<name>.json`: the swirl at its centre, and the overlay's shape and
colour) and the shared overlay code ([`overlay.js`](overlay.js)) at that
moment. When room is needed, overlays whose clip has been out of view for a
while are removed and their GPU resources freed. How many stay loaded follows
the device (`content: { budget: 'auto' }`): more on a laptop or fast phone,
fewer on a slow one, and fewer still while the frame rate suffers.

Tracking time is bounded too. When a camera frame is short of time, the clips
that matter less (smaller, cut off by the edge of the view, or away from the
centre) are tracked with fewer points, or updated only every few frames, and
the clip you point at keeps full resolution. The readout at the bottom of the
page shows each clip's priority, points and overlay state.

```sh
npm run build                      # builds dist/fastar.js
npx tsx scripts/example-multi.ts   # (re)generates clips/, and targets.bin here (needs ffmpeg)
npx serve .                        # serve the repository root
```

Open `http://localhost:3000/examples/multi/screen.html` on a computer and
click it to play the four clips full screen. Then open
`http://localhost:3000/examples/multi/` on a phone (over HTTPS, see
[the basic example](../basic/README.md)) and point it at the screen from about
30 cm, so the four clips fill the view. From further away each clip is too
small in the camera image to be found reliably. Move
closer to one clip and its tracking points go up while the others' go down.
Turn away from a clip for a few seconds and come back, and its overlay is
still there. With the overlays of other clips loaded meanwhile, it may be
loaded again.

`targets.bin` assumes each clip is shown about 14 cm wide (a quarter of a
laptop screen). Poses scale with that width, so for another screen size
recompile with the width in metres:

```sh
npx tsx cli/compile.ts --points 60 -o examples/multi/targets.bin examples/multi/clips/{amber,cobalt,jade,coral}.mp4:0.2
```
