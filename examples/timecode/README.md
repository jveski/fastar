# Audio timecode example

Phones listen for the set's timecode and show the set time. They flash on every whole second of it, so phones held side by side show at a glance whether they agree.

```sh
npm run build                                              # builds dist/fastar.js and the CLIs
node dist/fastar-timecode.js generate -o timecode.wav -d 1h  # an hour of timecode from set time 0
npx serve .                                                # serve the repository root
```

Play `timecode.wav` through speakers that reach 20 kHz (most studio monitors and PAs do; laptop and phone speakers mostly don't), at a level you can't hear. Open `http://localhost:3000/examples/timecode/` on a phone and tap Listen. Phones need HTTPS for the microphone: run `npx cloudflared tunnel --url http://localhost:3000` and open the printed `https://….trycloudflare.com/examples/timecode/` URL, or use any HTTPS host.

For an event, generate with its own key (`--key my-set`) and open the page with `?key=my-set`.

To check a venue, record on a phone (lossless: compressed recordings drop the band) and run `node dist/fastar-timecode.js decode recording.wav`.
