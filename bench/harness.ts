// Tiny benchmark harness with per-stage budgets.
import { PerformanceObserver, performance } from 'node:perf_hooks';

export interface BenchCase {
	name: string;
	/** Budget for the median iteration time in milliseconds. */
	budgetMs: number;
	/** Returns the function to time. Setup cost is excluded. */
	setup: () => () => void;
	iterations?: number;
	warmup?: number;
}

export interface BenchResult {
	name: string;
	medianMs: number;
	p95Ms: number;
	budgetMs: number;
	pass: boolean;
	gcCount: number;
	heapDeltaKB: number;
}

let gcEvents = 0;
const obs = new PerformanceObserver((list) => {
	gcEvents += list.getEntries().length;
});
obs.observe({ entryTypes: ['gc'] });

const flushObserver = () => new Promise<void>((r) => setTimeout(r, 0));

export async function runBench(c: BenchCase): Promise<BenchResult> {
	const fn = c.setup();
	const warm = c.warmup ?? 50;
	const n = c.iterations ?? 200;
	for (let i = 0; i < warm; i++) fn();
	(globalThis as { gc?: () => void }).gc?.();
	await flushObserver();
	const gc0 = gcEvents;
	const heap0 = process.memoryUsage().heapUsed;
	const times = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const t0 = performance.now();
		fn();
		times[i] = performance.now() - t0;
	}
	const heap1 = process.memoryUsage().heapUsed;
	await flushObserver();
	const gcCount = gcEvents - gc0;
	times.sort();
	const medianMs = times[n >> 1];
	const p95Ms = times[Math.min(n - 1, Math.floor(n * 0.95))];
	return {
		name: c.name,
		medianMs,
		p95Ms,
		budgetMs: c.budgetMs,
		pass: medianMs <= c.budgetMs,
		gcCount,
		heapDeltaKB: (heap1 - heap0) / 1024,
	};
}

export function printResults(results: BenchResult[]): boolean {
	const rows = results.map((r) => ({
		stage: r.name,
		'median ms': r.medianMs.toFixed(3),
		'p95 ms': r.p95Ms.toFixed(3),
		'budget ms': r.budgetMs.toFixed(2),
		gc: r.gcCount,
		'heap Δ KB': r.heapDeltaKB.toFixed(1),
		ok: r.pass ? '✓' : '✗',
	}));
	console.table(rows);
	return results.every((r) => r.pass);
}
