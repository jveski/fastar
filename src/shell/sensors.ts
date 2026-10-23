// Motion sensors for iOS Safari and Android Chrome, normalized to camera axes
// (OpenCV convention: x right, y down, z forward) and rad/s.
//
// Device axes (W3C): x right, y up (top of the device in its natural
// orientation), z out of the screen. Rear camera = (x, -y, -z) of screen
// axes; the front camera looks the other way and its raw image is mirrored:
// (-x, -y, z). Screen axes are device axes rotated by screen.orientation.angle.
import { RegularClock } from '../fusion/clock';

export type Facing = 'environment' | 'user';

export interface ImuReading {
  /** performance.now() milliseconds. */
  t: number;
  /** rad/s in camera axes. */
  gx: number;
  gy: number;
  gz: number;
  /** m/s^2 specific force (includes gravity) in camera axes, or NaN. */
  ax: number;
  ay: number;
  az: number;
}

export type SensorBackend = 'generic-sensor' | 'devicemotion' | 'none';

/** Outcome of the motion permission request ('not-required' off iOS). */
export type MotionPermission = 'not-required' | 'not-requested' | 'granted' | 'denied' | `error: ${string}`;

const DEG = Math.PI / 180;

/** Map a device-axes vector to camera axes for a screen rotation (deg) and camera facing. */
export function deviceToCamera(
  x: number, y: number, z: number, screenAngleDeg: number, facing: Facing, out: Float64Array | number[] = [0, 0, 0],
): Float64Array | number[] {
  const a = screenAngleDeg * DEG;
  const c = Math.cos(a), s = Math.sin(a);
  // Screen axes: x_s = (cos a, -sin a, 0), y_s = (sin a, cos a, 0) in device axes.
  const sx = c * x - s * y, sy = s * x + c * y;
  if (facing === 'environment') {
    out[0] = sx; out[1] = -sy; out[2] = -z;
  } else {
    out[0] = -sx; out[1] = -sy; out[2] = z;
  }
  return out;
}

export function screenAngle(): number {
  if (typeof screen !== 'undefined' && screen.orientation && typeof screen.orientation.angle === 'number') {
    return screen.orientation.angle;
  }
  const w = typeof window !== 'undefined' ? (window as unknown as { orientation?: number }).orientation : undefined;
  return typeof w === 'number' ? (w + 360) % 360 : 0;
}

interface GenericSensor extends EventTarget {
  x: number; y: number; z: number; timestamp: number;
  start(): void; stop(): void;
  onreading: (() => void) | null;
  onerror: ((e: Event & { error?: Error }) => void) | null;
}
type SensorCtor = new (o: { frequency: number; referenceFrame?: string }) => GenericSensor;

interface DeviceMotionEventStatic {
  requestPermission?: () => Promise<'granted' | 'denied'>;
}

export interface MotionSensorOptions {
  frequency?: number;
  facing?: Facing;
  /** Prefer devicemotion even when the Generic Sensor API exists. */
  forceDeviceMotion?: boolean;
}

/** Gyro + gravity source delivering camera-axis readings. */
export class MotionSensors {
  backend: SensorBackend = 'none';
  facing: Facing;
  private readonly frequency: number;
  private gyro: GenericSensor | null = null;
  private grav: GenericSensor | null = null;
  private motionHandler: ((e: DeviceMotionEvent) => void) | null = null;
  private readonly lastAccel = [NaN, NaN, NaN];
  private readonly g = [0, 0, 0];
  private readonly a = [0, 0, 0];

  constructor(private readonly onSample: (r: ImuReading) => void, private readonly o: MotionSensorOptions = {}) {
    this.frequency = o.frequency ?? 100;
    this.facing = o.facing ?? 'environment';
  }

  /** iOS 13+ requires a permission prompt from a user gesture. */
  static needsPermission(): boolean {
    const D = (globalThis as { DeviceMotionEvent?: DeviceMotionEventStatic }).DeviceMotionEvent;
    return !!D && typeof D.requestPermission === 'function';
  }

  /** Must be called from a user gesture on iOS. Resolves true if motion data is allowed. */
  static async requestPermission(): Promise<boolean> {
    const s = await MotionSensors.requestPermissionState();
    return s === 'granted' || s === 'not-required';
  }

  /** Like requestPermission(), but says why access was refused (denied, or the error thrown). */
  static async requestPermissionState(): Promise<MotionPermission> {
    const D = (globalThis as { DeviceMotionEvent?: DeviceMotionEventStatic }).DeviceMotionEvent;
    if (!D || typeof D.requestPermission !== 'function') return 'not-required';
    try {
      return (await D.requestPermission()) === 'granted' ? 'granted' : 'denied';
    } catch (e) {
      const err = e as Error;
      return `error: ${err?.name ?? 'Error'}: ${err?.message ?? String(e)}`;
    }
  }

  async start(): Promise<SensorBackend> {
    if (!this.o.forceDeviceMotion && (await this.startGeneric())) return (this.backend = 'generic-sensor');
    if (this.startDeviceMotion()) return (this.backend = 'devicemotion');
    return (this.backend = 'none');
  }

  stop(): void {
    this.gyro?.stop();
    this.grav?.stop();
    this.gyro = this.grav = null;
    if (this.motionHandler) window.removeEventListener('devicemotion', this.motionHandler);
    this.motionHandler = null;
    this.backend = 'none';
  }

  private async startGeneric(): Promise<boolean> {
    const G = globalThis as unknown as { Gyroscope?: SensorCtor; GravitySensor?: SensorCtor; Accelerometer?: SensorCtor };
    if (!G.Gyroscope) return false;
    try {
      const perms = navigator.permissions;
      if (perms) {
        const r = await perms.query({ name: 'gyroscope' as PermissionName }).catch(() => null);
        if (r && r.state === 'denied') return false;
      }
      const gyro = new G.Gyroscope({ frequency: this.frequency, referenceFrame: 'device' });
      const GravCtor = G.GravitySensor ?? G.Accelerometer;
      const grav = GravCtor ? new GravCtor({ frequency: Math.min(60, this.frequency), referenceFrame: 'device' }) : null;
      const ok = await new Promise<boolean>((resolve) => {
        let settled = false;
        const done = (v: boolean) => { if (!settled) { settled = true; resolve(v); } };
        gyro.onerror = () => done(false);
        gyro.onreading = () => {
          done(true);
          const ang = screenAngle();
          deviceToCamera(gyro.x, gyro.y, gyro.z, ang, this.facing, this.g);
          this.onSample({
            t: gyro.timestamp, gx: this.g[0], gy: this.g[1], gz: this.g[2],
            ax: this.lastAccel[0], ay: this.lastAccel[1], az: this.lastAccel[2],
          });
        };
        setTimeout(() => done(false), 1000);
        gyro.start();
      });
      if (!ok) { gyro.stop(); return false; }
      if (grav) {
        grav.onreading = () => {
          deviceToCamera(grav.x, grav.y, grav.z, screenAngle(), this.facing, this.a);
          this.lastAccel[0] = this.a[0]; this.lastAccel[1] = this.a[1]; this.lastAccel[2] = this.a[2];
        };
        grav.onerror = () => {};
        grav.start();
      }
      this.gyro = gyro;
      this.grav = grav;
      return true;
    } catch {
      return false;
    }
  }

  private startDeviceMotion(): boolean {
    if (typeof window === 'undefined' || !('DeviceMotionEvent' in window)) return false;
    // Event stamps are dispatch times (bursty, 0-30 ms apart at 60 Hz), so
    // put the samples back on their regular grid. Events are queued, not
    // dropped, so every event is the next sample.
    let clock: RegularClock | null = null;
    this.motionHandler = (e: DeviceMotionEvent) => {
      const r = e.rotationRate;
      if (!r || r.alpha === null || r.beta === null || r.gamma === null) return;
      if (!clock) {
        // interval: ms per spec; WebKit reports seconds.
        const iv = e.interval > 0 ? (e.interval < 1 ? e.interval * 1000 : e.interval) : 1000 / 60;
        clock = new RegularClock(iv, Infinity);
      }
      const ang = screenAngle();
      // W3C DeviceMotionEventRotationRate: alpha about x, beta about y, gamma
      // about z (device axes), in deg/s. Not the Z-X'-Y'' order of the
      // orientation angles, though the spec's own vehicle example still uses it.
      deviceToCamera(r.alpha * DEG, r.beta * DEG, r.gamma * DEG, ang, this.facing, this.g);
      const acc = e.accelerationIncludingGravity;
      let ax = NaN, ay = NaN, az = NaN;
      if (acc && acc.x !== null && acc.y !== null && acc.z !== null) {
        deviceToCamera(acc.x, acc.y, acc.z, ang, this.facing, this.a);
        ax = this.a[0]; ay = this.a[1]; az = this.a[2];
      }
      this.onSample({ t: clock.map(e.timeStamp), gx: this.g[0], gy: this.g[1], gz: this.g[2], ax, ay, az });
    };
    window.addEventListener('devicemotion', this.motionHandler);
    return true;
  }
}
