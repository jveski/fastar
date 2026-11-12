/// <reference lib="webworker" />
// Worker entry: one script, three roles (tracking, acquisition or timecode),
// chosen by the first message.
import { TrackerCore } from './tracker-core';
import { AcquireCore } from './acquire-core';
import { TimecodeCore, type TimecodeCalibrate, type TimecodeInit } from './timecode-core';
import type { AcquireInit, ToTracker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let tracker: TrackerCore | null = null;
let timecode: TimecodeCore | null = null;

self.onmessage = (e: MessageEvent) => {
  const m = e.data as ToTracker | AcquireInit | TimecodeInit | TimecodeCalibrate;
  if (m.type === 'init-acquire') {
    try {
      new AcquireCore().init(m);
      self.postMessage({ type: 'acquire-ready' });
    } catch (e) {
      self.postMessage({ type: 'acquire-error', message: (e as Error).message ?? String(e) });
    }
    return;
  }
  if (m.type === 'init-timecode') {
    timecode = new TimecodeCore((msg) => self.postMessage(msg));
    timecode.init(m);
    return;
  }
  if (m.type === 'calibrate') {
    timecode?.handle(m);
    return;
  }
  tracker ??= new TrackerCore((msg, transfer) => self.postMessage(msg, transfer ?? []));
  tracker.handle(m);
};
