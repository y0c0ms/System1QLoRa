import fs from 'node:fs';
import path from 'node:path';

const R = path.resolve(process.cwd(), '..', 'results');

type EvalDataset = {
  T: number;
  n_test: number;
  skipped_gt26: number;
  acc_all: number;
  acc_by_task: Record<string, number>;
};
export type EvalReport = {
  base: string;
  adapter: string;
  datasets: Record<string, EvalDataset>;
};
export type HardCase = {
  tag: string;
  state: string;
  question: string;
  options: string[];
  probs: number[];
  pick: number;
  expect: number;
  correct: boolean;
  latency_ms: number;
};
export type HardResults = { n_correct: number; n_total: number; cases: HardCase[] };

// Our own generated files with a known shape → cast the parsed value to the domain type.
function readJSON<T>(p: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8')) as T;
  } catch {
    return null;
  }
}

export const evalReport = readJSON<EvalReport>(`${R}/jevlike4b_eval.json`);
export const hardResults = readJSON<HardResults>(`${R}/demo_hard_results.json`);

// Comparison table (numbers from docs/FINDINGS.md, all calibrated test accuracy)
export const benchmarks = [
  {
    name: 'SNI held-out-predicate',
    n: 1950,
    floor: 'majority 0.534',
    priorOpen: 0.472,
    prompted35b: null,
    trained270m: null,
    ours4b: evalReport?.datasets?.sni?.acc_all ?? 0.672,
    note: 'Generality benchmark that broke every small scorer',
  },
  {
    name: 'reflex (math + code)',
    n: 600,
    floor: '\u2014',
    priorOpen: null,
    prompted35b: 0.385,
    trained270m: 0.530,
    ours4b: evalReport?.datasets?.reflex?.acc_all ?? 0.580,
    note: 'Beats both the prompted 35B and the trained 270M',
  },
  {
    name: 'BFCL tool-calling',
    n: 399,
    floor: '\u2014',
    priorOpen: null,
    prompted35b: 0.9373,
    trained270m: null,
    ours4b: evalReport?.datasets?.bfcl?.acc_all ?? 0.937,
    note: 'Matches the 35B \u2014 and ZERO-SHOT (held out of training)',
  },
];

export const reflexPerTask = [
  { task: 'math_topic', b35: 0.485, m270: 0.695, b4: evalReport?.datasets?.reflex?.acc_by_task?.math_topic ?? 0.785 },
  { task: 'math_level', b35: 0.220, m270: 0.345, b4: evalReport?.datasets?.reflex?.acc_by_task?.math_level ?? 0.410 },
  { task: 'code_defect', b35: 0.450, m270: 0.550, b4: evalReport?.datasets?.reflex?.acc_by_task?.code_defect ?? 0.545 },
];

export const model = {
  base: 'Qwen3-4B-Instruct-2507',
  method: 'QLoRA (4-bit NF4, r=16, ~33M trainable)',
  corpus: 'SNI train + reflex train = 22,185 rows',
  hardware: 'RX 7900 GRE 16 GB, ~4h',
};
