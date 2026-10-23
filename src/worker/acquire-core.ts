// Acquisition worker logic: detect + match on request, never blocking tracking.
import { decodeTargets } from '../target/format';
import { Acquirer } from '../track/acquire';
import type { AcquireInit, AcquireToTracker, TrackerToAcquire, WireAcquireResult } from './protocol';
import type { PortLike } from './tracker-core';

export class AcquireCore {
  private acq: Acquirer | null = null;
  private port: PortLike | null = null;

  constructor(private readonly now: () => number = () => performance.now()) {}

  init(m: AcquireInit): void {
    this.acq = new Acquirer(decodeTargets(m.targets), m.width, m.height, m.K);
    this.port = m.port as unknown as PortLike;
    this.port.onmessage = (e: MessageEvent) => this.handle(e.data as TrackerToAcquire);
  }

  handle(m: TrackerToAcquire): void {
    const acq = this.acq;
    if (!acq) return;
    if (m.type === 'intrinsics') { acq.K = m.K; return; }
    const t0 = this.now();
    const r = acq.acquire(new Uint8Array(m.luma), { exclude: m.exclude, up: m.up });
    let result: WireAcquireResult | null = null;
    if (r) result = { target: r.target, R: r.R, t: r.t, H: r.H, inliers: r.inliers, cov: r.cov, frame: r.frame, tgt: r.tgt };
    const msg: AcquireToTracker = { type: 'acquired', frameId: m.frameId, result, ms: this.now() - t0 };
    this.port!.postMessage(msg);
  }
}
