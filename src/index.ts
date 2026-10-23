export const VERSION = '0.1.0';

export { ImageTracker } from './shell/image-tracker';
export type { CreateOptions, StartOptions, Pose, PresentedFrame, FrameEvent, TrackerStats } from './shell/image-tracker';
export type { SmoothingOptions } from './fusion/smooth';
export { Recorder, sessionFiles } from './shell/recorder';
export type { RecordedSession, SessionLog, RecordOptions, RecordedFrame } from './shell/recorder';
export { MotionSensors, deviceToCamera } from './shell/sensors';
export type { ImuReading, Facing, SensorBackend, MotionPermission } from './shell/sensors';
export { ThreeAdapter } from './adapters/three';
export type { ThreeAdapterOptions } from './adapters/three';
export { registerAFrame } from './adapters/aframe';
export { compileImage, compileImages, imageToGray, type ImageInput } from './shell/compile';

export { compileTarget, compileTargets } from './target/compiler';
export type { CompileOptions } from './target/compiler';
export { encodeTargets, decodeTargets } from './target/format';
export type { TargetData, TargetMeta, QualityReport } from './target/format';
export { rgbaToGray, type GrayImage } from './image/gray';
export { glProjection, poseToGl, intrinsicsFromFov, type Intrinsics, type FitMode } from './vision/camera';
