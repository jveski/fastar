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
    try {
      new AcquireCore().init(m);
      self.postMessage({ type: 'acquire-ready' });
    } catch (e) {
      self.postMessage({ type: 'acquire-error', message: (e as Error).message ?? String(e) });
    }
    return;
  }
  tracker ??= new TrackerCore((msg, transfer) => self.postMessage(msg, transfer ?? []));
  tracker.handle(m);
};
