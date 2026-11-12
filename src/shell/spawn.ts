// Worker creation: from the inlined worker source, or a URL (strict CSP).
import workerSource from 'virtual:worker-source';

export function spawnWorker(url?: string | URL): Worker {
  if (url) return new Worker(url);
  const blob = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
  const w = new Worker(blob);
  setTimeout(() => URL.revokeObjectURL(blob), 10000);
  return w;
}
