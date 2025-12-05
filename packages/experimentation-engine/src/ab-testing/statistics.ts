/**
 * NEURECTOMY A/B Testing Statistics
 * @module @neurectomy/experimentation-engine/ab-testing
 * @agent @PRISM @FLUX
 *
 * Statistical analysis for A/B tests including hypothesis testing,
 * confidence intervals, sample size calculations, and Bayesian analysis.
 */

// ============================================================================
// Types
// ============================================================================

export interface StatisticalTestResult {
  testType: string;
  pValue: number;
  statistic: number;
  significant: boolean;
  confidenceLevel: number;
  effectSize: number;
  confidenceInterval: [number, number];
  sampleSizes: { control: number; treatment: number };
  power?: number;
}

export interface BayesianResult {
  probabilityBBeatsA: number;
  probabilityABeatsB: number;
  expectedLoss: { control: number; treatment: number };
  credibleInterval: [number, number];
  posteriorMean: { control: number; treatment: number };
  posteriorVariance: { control: number; treatment: number };
}

export interface SampleSizeResult {
  perVariant: number;
  total: number;
  parameters: {
    baselineRate: number;
    minimumDetectableEffect: number;
    confidenceLevel: number;
    power: number;
  };
}

export interface SequentialTestResult {
  decision: "stop_winner" | "stop_loser" | "continue";
  winner?: "control" | "treatment";
  boundaryReached: boolean;
  zScore: number;
  boundary: number;
  adjustedPValue?: number;
}

export interface MultipleTestCorrection {
  method: "bonferroni" | "holm" | "benjamini-hochberg" | "none";
  originalPValues: number[];
  adjustedPValues: number[];
  significantTests: boolean[];
}

// ============================================================================
// Statistical Functions
// ============================================================================

/**
 * Standard normal cumulative distribution function
 */
function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);

  const t = 1.0 / (1.0 + p * x);
  const y =
    1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1.0 + sign * y);
}

/**
 * Inverse of the standard normal CDF (quantile function)
 */
function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  // Rational approximation for lower region
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number, r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}

/**
 * Student's t-distribution CDF approximation
 */
function tCDF(t: number, df: number): number {
  const x = df / (df + t * t);
  const prob = 0.5 * incompleteBeta(x, df / 2, 0.5);
  return t >= 0 ? 1 - prob : prob;
}

/**
 * Incomplete beta function approximation
 */
function incompleteBeta(x: number, a: number, b: number): number {
  if (x === 0 || x === 1) return x;

  const maxIterations = 200;
  const epsilon = 1e-15;

  // Use continued fraction expansion
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;

  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIterations; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = d * c;
    h *= delta;

    if (Math.abs(delta - 1) < epsilon) break;
  }

  const bt = Math.exp(
    logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log(1 - x)
  );

  return (bt * h) / a;
}

/**
 * Log gamma function approximation (Lanczos)
 */
function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }

  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;

  for (let i = 1; i < g + 2; i++) {
    a += c[i] / (x + i);
  }

  return (
    0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
  );
}

/**
 * Beta function
 */
function beta(a: number, b: number): number {
  return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b));
}

// ============================================================================
// Frequentist Tests
// ============================================================================

/**
 * Two-proportion Z-test for conversion rates
 */
export function proportionZTest(
  controlConversions: number,
  controlTotal: number,
  treatmentConversions: number,
  treatmentTotal: number,
  confidenceLevel: number = 0.95
): StatisticalTestResult {
  const p1 = controlConversions / controlTotal;
  const p2 = treatmentConversions / treatmentTotal;

  // Pooled proportion
  const pooled =
    (controlConversions + treatmentConversions) /
    (controlTotal + treatmentTotal);

  // Standard error
  const se = Math.sqrt(
    pooled * (1 - pooled) * (1 / controlTotal + 1 / treatmentTotal)
  );

  // Z-statistic
  const z = (p2 - p1) / se;

  // P-value (two-tailed)
  const pValue = 2 * (1 - normalCDF(Math.abs(z)));

  // Effect size (relative difference)
  const effectSize = p1 > 0 ? (p2 - p1) / p1 : 0;

  // Confidence interval for difference
  const alpha = 1 - confidenceLevel;
  const zCrit = normalQuantile(1 - alpha / 2);
  const seDiff = Math.sqrt(
    (p1 * (1 - p1)) / controlTotal + (p2 * (1 - p2)) / treatmentTotal
  );
  const ci: [number, number] = [
    p2 - p1 - zCrit * seDiff,
    p2 - p1 + zCrit * seDiff,
  ];

  return {
    testType: "two-proportion-z-test",
    pValue,
    statistic: z,
    significant: pValue < alpha,
    confidenceLevel,
    effectSize,
    confidenceInterval: ci,
    sampleSizes: { control: controlTotal, treatment: treatmentTotal },
  };
}

/**
 * Two-sample t-test for continuous metrics
 */
export function twoSampleTTest(
  control: { mean: number; variance: number; n: number },
  treatment: { mean: number; variance: number; n: number },
  confidenceLevel: number = 0.95
): StatisticalTestResult {
  const { mean: m1, variance: v1, n: n1 } = control;
  const { mean: m2, variance: v2, n: n2 } = treatment;

  // Welch's t-test (unequal variances)
  const se = Math.sqrt(v1 / n1 + v2 / n2);
  const t = (m2 - m1) / se;

  // Welch-Satterthwaite degrees of freedom
  const df =
    Math.pow(v1 / n1 + v2 / n2, 2) /
    (Math.pow(v1 / n1, 2) / (n1 - 1) + Math.pow(v2 / n2, 2) / (n2 - 1));

  // P-value (two-tailed)
  const pValue = 2 * (1 - tCDF(Math.abs(t), df));

  // Effect size (Cohen's d)
  const pooledStd = Math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2));
  const effectSize = pooledStd > 0 ? (m2 - m1) / pooledStd : 0;

  // Confidence interval for difference
  const alpha = 1 - confidenceLevel;
  const tCrit = tQuantile(1 - alpha / 2, df);
  const ci: [number, number] = [m2 - m1 - tCrit * se, m2 - m1 + tCrit * se];

  return {
    testType: "welch-t-test",
    pValue,
    statistic: t,
    significant: pValue < alpha,
    confidenceLevel,
    effectSize,
    confidenceInterval: ci,
    sampleSizes: { control: n1, treatment: n2 },
  };
}

/**
 * Approximate t-distribution quantile
 */
function tQuantile(p: number, df: number): number {
  // Use normal approximation for large df
  if (df > 100) {
    return normalQuantile(p);
  }

  // Newton-Raphson iteration
  let x = normalQuantile(p);
  for (let i = 0; i < 10; i++) {
    const fx = tCDF(x, df) - p;
    const dx = 0.0001;
    const fpx = (tCDF(x + dx, df) - tCDF(x - dx, df)) / (2 * dx);
    if (Math.abs(fpx) < 1e-10) break;
    x = x - fx / fpx;
  }
  return x;
}

/**
 * Chi-squared test for categorical data
 */
export function chiSquaredTest(
  observed: number[][],
  confidenceLevel: number = 0.95
): StatisticalTestResult {
  const rows = observed.length;
  const cols = observed[0].length;

  // Calculate row and column totals
  const rowTotals = observed.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = observed[0].map((_, j) =>
    observed.reduce((sum, row) => sum + row[j], 0)
  );
  const total = rowTotals.reduce((a, b) => a + b, 0);

  // Calculate expected values and chi-squared statistic
  let chiSq = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const expected = (rowTotals[i] * colTotals[j]) / total;
      if (expected > 0) {
        chiSq += Math.pow(observed[i][j] - expected, 2) / expected;
      }
    }
  }

  const df = (rows - 1) * (cols - 1);
  const pValue = 1 - chiSquaredCDF(chiSq, df);
  const alpha = 1 - confidenceLevel;

  // Cramer's V effect size
  const minDim = Math.min(rows - 1, cols - 1);
  const effectSize = minDim > 0 ? Math.sqrt(chiSq / (total * minDim)) : 0;

  return {
    testType: "chi-squared-test",
    pValue,
    statistic: chiSq,
    significant: pValue < alpha,
    confidenceLevel,
    effectSize,
    confidenceInterval: [0, 0], // Not applicable for chi-squared
    sampleSizes: { control: total, treatment: 0 },
  };
}

/**
 * Chi-squared CDF approximation
 */
function chiSquaredCDF(x: number, df: number): number {
  if (x <= 0) return 0;
  return gammaCDF(x / 2, df / 2);
}

/**
 * Gamma CDF approximation
 */
function gammaCDF(x: number, a: number): number {
  if (x <= 0) return 0;
  if (x < a + 1) {
    return gammaIncLower(a, x) / Math.exp(logGamma(a));
  }
  return 1 - gammaIncUpper(a, x) / Math.exp(logGamma(a));
}

/**
 * Lower incomplete gamma function
 */
function gammaIncLower(a: number, x: number): number {
  let sum = 0;
  let term = 1 / a;
  for (let n = 1; n <= 100; n++) {
    sum += term;
    term *= x / (a + n);
    if (Math.abs(term) < 1e-10) break;
  }
  return Math.pow(x, a) * Math.exp(-x) * sum;
}

/**
 * Upper incomplete gamma function (continued fraction)
 */
function gammaIncUpper(a: number, x: number): number {
  let f = 1e-30;
  let c = 1e-30;
  let d = 0;

  for (let i = 1; i <= 100; i++) {
    const an = i * (a - i);
    const bn = x + 2 * i + 1 - a;
    d = bn + an * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = bn + an / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-10) break;
  }

  return Math.pow(x, a) * Math.exp(-x) * f;
}

// ============================================================================
// Bayesian Analysis
// ============================================================================

/**
 * Bayesian A/B test for conversion rates
 */
export function bayesianABTest(
  controlConversions: number,
  controlTotal: number,
  treatmentConversions: number,
  treatmentTotal: number,
  priorAlpha: number = 1,
  priorBeta: number = 1,
  samples: number = 10000
): BayesianResult {
  // Posterior parameters (Beta distribution)
  const alphaA = priorAlpha + controlConversions;
  const betaA = priorBeta + controlTotal - controlConversions;
  const alphaB = priorAlpha + treatmentConversions;
  const betaB = priorBeta + treatmentTotal - treatmentConversions;

  // Monte Carlo sampling
  let bBeatsA = 0;
  let lossA = 0;
  let lossB = 0;

  for (let i = 0; i < samples; i++) {
    const sampleA = betaSample(alphaA, betaA);
    const sampleB = betaSample(alphaB, betaB);

    if (sampleB > sampleA) {
      bBeatsA++;
      lossA += sampleB - sampleA;
    } else {
      lossB += sampleA - sampleB;
    }
  }

  const probBBeatsA = bBeatsA / samples;
  const expectedLossA = lossA / samples;
  const expectedLossB = lossB / samples;

  // Posterior mean and variance
  const posteriorMeanA = alphaA / (alphaA + betaA);
  const posteriorMeanB = alphaB / (alphaB + betaB);
  const posteriorVarA =
    (alphaA * betaA) / (Math.pow(alphaA + betaA, 2) * (alphaA + betaA + 1));
  const posteriorVarB =
    (alphaB * betaB) / (Math.pow(alphaB + betaB, 2) * (alphaB + betaB + 1));

  // 95% credible interval for difference
  const diffSamples: number[] = [];
  for (let i = 0; i < samples; i++) {
    diffSamples.push(betaSample(alphaB, betaB) - betaSample(alphaA, betaA));
  }
  diffSamples.sort((a, b) => a - b);
  const ciLow = diffSamples[Math.floor(samples * 0.025)];
  const ciHigh = diffSamples[Math.floor(samples * 0.975)];

  return {
    probabilityBBeatsA: probBBeatsA,
    probabilityABeatsB: 1 - probBBeatsA,
    expectedLoss: { control: expectedLossA, treatment: expectedLossB },
    credibleInterval: [ciLow, ciHigh],
    posteriorMean: { control: posteriorMeanA, treatment: posteriorMeanB },
    posteriorVariance: { control: posteriorVarA, treatment: posteriorVarB },
  };
}

/**
 * Sample from Beta distribution
 */
function betaSample(alpha: number, beta: number): number {
  const x = gammaSample(alpha);
  const y = gammaSample(beta);
  return x / (x + y);
}

/**
 * Sample from Gamma distribution (Marsaglia & Tsang)
 */
function gammaSample(shape: number): number {
  if (shape < 1) {
    return gammaSample(1 + shape) * Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x: number;
    let v: number;
    do {
      x = normalSample();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.random();
    const x2 = x * x;

    if (u < 1 - 0.0331 * x2 * x2) {
      return d * v;
    }

    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}

/**
 * Sample from standard normal distribution (Box-Muller)
 */
function normalSample(): number {
  let u: number, v: number, s: number;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);

  return u * Math.sqrt((-2 * Math.log(s)) / s);
}

// ============================================================================
// Sample Size Calculations
// ============================================================================

/**
 * Calculate required sample size for A/B test
 */
export function calculateSampleSize(
  baselineRate: number,
  minimumDetectableEffect: number,
  confidenceLevel: number = 0.95,
  power: number = 0.8
): SampleSizeResult {
  const alpha = 1 - confidenceLevel;
  const zAlpha = normalQuantile(1 - alpha / 2);
  const zBeta = normalQuantile(power);

  const p1 = baselineRate;
  const p2 = baselineRate * (1 + minimumDetectableEffect);
  const pBar = (p1 + p2) / 2;

  const n =
    Math.pow(
      zAlpha * Math.sqrt(2 * pBar * (1 - pBar)) +
        zBeta * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)),
      2
    ) / Math.pow(p2 - p1, 2);

  const perVariant = Math.ceil(n);

  return {
    perVariant,
    total: perVariant * 2,
    parameters: {
      baselineRate,
      minimumDetectableEffect,
      confidenceLevel,
      power,
    },
  };
}

/**
 * Calculate statistical power given sample sizes
 */
export function calculatePower(
  baselineRate: number,
  minimumDetectableEffect: number,
  sampleSizePerVariant: number,
  confidenceLevel: number = 0.95
): number {
  const alpha = 1 - confidenceLevel;
  const zAlpha = normalQuantile(1 - alpha / 2);

  const p1 = baselineRate;
  const p2 = baselineRate * (1 + minimumDetectableEffect);
  const pBar = (p1 + p2) / 2;
  const n = sampleSizePerVariant;

  const se1 = Math.sqrt((2 * pBar * (1 - pBar)) / n);
  const se2 = Math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n);

  const zBeta = (Math.abs(p2 - p1) - zAlpha * se1) / se2;

  return normalCDF(zBeta);
}

// ============================================================================
// Sequential Testing
// ============================================================================

/**
 * O'Brien-Fleming sequential testing boundary
 */
export function sequentialTest(
  controlConversions: number,
  controlTotal: number,
  treatmentConversions: number,
  treatmentTotal: number,
  lookNumber: number,
  totalLooks: number,
  alpha: number = 0.05
): SequentialTestResult {
  // O'Brien-Fleming spending function
  const spentAlpha =
    2 *
    (1 -
      normalCDF(
        normalQuantile(1 - alpha / 2) / Math.sqrt(lookNumber / totalLooks)
      ));

  // Calculate Z-score
  const p1 = controlConversions / controlTotal;
  const p2 = treatmentConversions / treatmentTotal;
  const pooled =
    (controlConversions + treatmentConversions) /
    (controlTotal + treatmentTotal);
  const se = Math.sqrt(
    pooled * (1 - pooled) * (1 / controlTotal + 1 / treatmentTotal)
  );
  const z = (p2 - p1) / se;

  // O'Brien-Fleming boundary
  const boundary = normalQuantile(1 - spentAlpha / 2);

  const boundaryReached = Math.abs(z) >= boundary;

  let decision: SequentialTestResult["decision"];
  let winner: "control" | "treatment" | undefined;

  if (boundaryReached) {
    if (z > 0) {
      decision = "stop_winner";
      winner = "treatment";
    } else {
      decision = "stop_loser";
      winner = "control";
    }
  } else {
    decision = "continue";
  }

  return {
    decision,
    winner,
    boundaryReached,
    zScore: z,
    boundary,
    adjustedPValue: 2 * (1 - normalCDF(Math.abs(z))),
  };
}

// ============================================================================
// Multiple Testing Correction
// ============================================================================

/**
 * Apply multiple testing correction
 */
export function multipleTestCorrection(
  pValues: number[],
  method: MultipleTestCorrection["method"] = "benjamini-hochberg",
  alpha: number = 0.05
): MultipleTestCorrection {
  const n = pValues.length;
  const indices = pValues.map((_, i) => i);
  const sorted = indices.sort((a, b) => pValues[a] - pValues[b]);
  const adjustedPValues = new Array(n).fill(1);

  switch (method) {
    case "bonferroni":
      for (let i = 0; i < n; i++) {
        adjustedPValues[i] = Math.min(1, pValues[i] * n);
      }
      break;

    case "holm":
      for (let i = 0; i < n; i++) {
        const rank = sorted.indexOf(i);
        adjustedPValues[i] = Math.min(1, pValues[i] * (n - rank));
      }
      // Enforce monotonicity
      for (let i = 1; i < n; i++) {
        const idx = sorted[i];
        const prevIdx = sorted[i - 1];
        adjustedPValues[idx] = Math.max(
          adjustedPValues[idx],
          adjustedPValues[prevIdx]
        );
      }
      break;

    case "benjamini-hochberg":
      for (let i = n - 1; i >= 0; i--) {
        const idx = sorted[i];
        const rank = i + 1;
        adjustedPValues[idx] = (pValues[idx] * n) / rank;
        if (i < n - 1) {
          const nextIdx = sorted[i + 1];
          adjustedPValues[idx] = Math.min(
            adjustedPValues[idx],
            adjustedPValues[nextIdx]
          );
        }
        adjustedPValues[idx] = Math.min(1, adjustedPValues[idx]);
      }
      break;

    case "none":
    default:
      for (let i = 0; i < n; i++) {
        adjustedPValues[i] = pValues[i];
      }
  }

  const significantTests = adjustedPValues.map((p) => p < alpha);

  return {
    method,
    originalPValues: pValues,
    adjustedPValues,
    significantTests,
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate minimum detectable effect
 */
export function minimumDetectableEffect(
  baselineRate: number,
  sampleSizePerVariant: number,
  confidenceLevel: number = 0.95,
  power: number = 0.8
): number {
  const alpha = 1 - confidenceLevel;
  const zAlpha = normalQuantile(1 - alpha / 2);
  const zBeta = normalQuantile(power);
  const n = sampleSizePerVariant;

  // Approximation
  const se = Math.sqrt((2 * baselineRate * (1 - baselineRate)) / n);
  const mde = ((zAlpha + zBeta) * se) / baselineRate;

  return mde;
}

/**
 * Calculate lift (relative change)
 */
export function calculateLift(
  controlRate: number,
  treatmentRate: number
): { lift: number; relative: boolean } {
  if (controlRate === 0) {
    return { lift: treatmentRate > 0 ? Infinity : 0, relative: false };
  }
  return {
    lift: (treatmentRate - controlRate) / controlRate,
    relative: true,
  };
}
