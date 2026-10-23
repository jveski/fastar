/// <reference lib="webworker" />
// Worker entry: one script, two roles (tracking or acquisition), chosen by
// the first message.
import { TrackerCore } from './tracker-core';
import { AcquireCore } from './acquire-core';
import type { AcquireInit, ToTracker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let tracker: TrackerCore | null = null;

self.onmessage = (e: MessageEvent) => {
  const m = e.data as ToTracker | AcquireInit;
  if (m.type === 'init-acquire') {
    new AcquireCore().init(m);
    return;
  }
  tracker ??= new TrackerCore((msg, transfer) => self.postMessage(msg, transfer ?? []));
  tracker.handle(m);
};
