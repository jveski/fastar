# Browser E2E suite

This suite runs the **built** `dist/fastar.js` in real browser engines. It is
separate from the fast Node/Vitest suite (`npm test`) and is not run on pull
requests. Nightly and manual runs live in `.github/workflows/browser.yml`.

## Local

```sh
npm run test:browser
```

That builds `dist/fastar.js`, bundles the harness, regenerates the fake-camera
Y4M from the committed sweep clip, and runs Playwright. Requires `ffmpeg` for
the fake-camera fixture and Playwright's browsers:

```sh
npx playwright install chromium firefox webkit
```

Useful narrower runs:

```sh
npx playwright test --project=chromium-desktop
npx playwright test --project=mobile-chrome -g "live canvas"
npx playwright test --project=camera-chromium
```

The camera project and the `forceCopyPath` / no-WebGL tests are Chromium-only.
Firefox's Playwright build may fail to launch inside sandboxed local shells;
CI is the source of truth for Firefox.

The GitHub workflow only runs nightly and from **Run workflow**; it has no
pull-request trigger. GitHub only exposes that manual button after the workflow
exists on the repository's default branch.

## What is covered

- **Replay** of deterministic rendered scenes (`static`, `sweep`, `occlusion`,
  `video`) through `start({ replay: { log, video } })`, with per-frame ground
  truth and overlay accuracy assertions.
- **Live stream** input from a canvas `captureStream`, including `syncDisplay`.
- **`getUserMedia`** via Chromium's fake camera file (`camera-chromium`).
- **Sensors**: synthetic `devicemotion` fed through `MotionSensors`, plus the
  vision-only permission-denied path.
- **Recorder round trip**: record a live canvas stream with `MediaRecorder`,
  then replay the produced session in the same page.
- **Mobile emulation**: Pixel 8 Chrome and iPhone 15 Safari viewports,
  including a mid-run orientation/viewport change.
- **Android emulator** Chrome (Appium/UiAutomator2) and **iOS Simulator**
  Mobile Safari (Appium/XCUITest). Both are CI-only; see `runners/`.
- **Real-device golden sessions** in `fixtures/golden/` (skipped until a
  recording from a physical phone is added).

## Fixtures

`fixtures/data/` is committed. Regenerate it after changing the simulator,
camera model, or target poster:

```sh
npm run fixtures:browser
```

This produces:

| File | Contents |
| --- | --- |
| `*.mp4` | rendered camera feed at 640×480, 30 fps |
| `*.session.json` | `SessionLog` (frame/IMU timing) for replay |
| `*.truth.json` | per-frame target pose, visibility, video time |
| `targets-image.bin` | compiled poster target |
| `targets-video.bin` | compiled motion-clip target |
| `target.png` | poster target for inspection |

The harness pins `hfov: 63` because the fixtures use that field of view. Real
phones use the user-agent lookup table in `src/vision/camera.ts`.

## Real-device recordings

The golden tier requires files recorded on physical iPhones/Android phones;
see `fixtures/golden/README.md`. The simulator cannot fabricate camera or IMU
goldens, so the spec skips while that directory is empty.
