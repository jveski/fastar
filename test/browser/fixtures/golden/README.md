# Real-device golden sessions

Put short recordings from physical phones here. Each session needs three files
with the same base name:

```text
<name>.session.json   SessionLog downloaded by the bench page's Record button
<name>.mp4             the matching video
<name>.bin             the compiled target used in the recording
```

The browser suite replays these recordings through the shipped bundle. They
are the only tests that exercise real camera and IMU data.

To record one:

1. Open `dist/bench/` (or the bench page on a local HTTPS server) on the phone.
2. Load the target, tap Start, and record for 3–5 seconds while the target is
   visible. Include a moment where it leaves the frame if possible.
3. Use **Record** (video + JSON); the luma option is not needed here.
4. Rename the downloads to a single base name and copy them here, together
   with the target `.bin`.

Keep clips short (under ~5 MB). This directory may stay empty; the golden spec
skips when there are no sessions.
