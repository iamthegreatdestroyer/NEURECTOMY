/**
 * Hybrid Reality Twins POC
 *
 * Implements bidirectional synchronization between digital twins and physical
 * systems using sensor fusion, predictive synchronization, and latency compensation.
 *
 * Key Innovations:
 * - Multi-sensor fusion with Kalman filtering
 * - Predictive synchronization for latency compensation
 * - Bidirectional reality bridge (digital ↔ physical)
 * - Adaptive synchronization with drift correction
 *
 * Research Foundations:
 * - Kalman (1960): A New Approach to Linear Filtering and Prediction
 * - Grieves (2014): Digital Twin: Manufacturing Excellence through Virtual Factory Replication
 * - Tao et al. (2019): Digital Twins and Cyber-Physical Systems
 *
 * @elite-agents @VELOCITY @SYNAPSE @APEX
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type Timestamp = number;
type SensorId = string;
type EntityId = string;

interface Vector3D {
  x: number;
  y: number;
  z: number;
}

interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

interface PhysicalState {
  entityId: EntityId;
  timestamp: Timestamp;
  position: Vector3D;
  velocity: Vector3D;
  orientation: Quaternion;
  properties: Map<string, number>;
}

interface DigitalState {
  entityId: EntityId;
  timestamp: Timestamp;
  position: Vector3D;
  velocity: Vector3D;
  orientation: Quaternion;
  predictedPosition: Vector3D;
  confidence: number;
}

interface SensorReading {
  sensorId: SensorId;
  timestamp: Timestamp;
  value: Vector3D;
  accuracy: number;
  latency: number;
}

interface FusedState {
  position: Vector3D;
  velocity: Vector3D;
  covariance: number[][];
  confidence: number;
  timestamp: Timestamp;
}

interface SyncMetrics {
  latency: number;
  drift: number;
  confidence: number;
  lastSync: Timestamp;
}

// ============================================================================
// Sensor Fusion with Kalman Filter
// ============================================================================

class SensorFusion {
  private state: number[]; // State vector [x, y, z, vx, vy, vz]
  private covariance: number[][]; // State covariance matrix
  private processNoise: number;
  private measurementNoise: number;
  private dt: number;

  constructor(
    initialState: Vector3D,
    initialVelocity: Vector3D = { x: 0, y: 0, z: 0 },
    processNoise: number = 0.1,
    measurementNoise: number = 0.5
  ) {
    this.state = [
      initialState.x,
      initialState.y,
      initialState.z,
      initialVelocity.x,
      initialVelocity.y,
      initialVelocity.z,
    ];

    // Initialize covariance as identity
    this.covariance = this.createIdentityMatrix(6);
    this.processNoise = processNoise;
    this.measurementNoise = measurementNoise;
    this.dt = 0.016; // 60 FPS default
  }

  /**
   * Predict step: propagate state forward
   */
  predict(dt: number): void {
    this.dt = dt;

    // State transition matrix (constant velocity model)
    const F = [
      [1, 0, 0, dt, 0, 0],
      [0, 1, 0, 0, dt, 0],
      [0, 0, 1, 0, 0, dt],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
    ];

    // Predict state: x = F * x
    this.state = this.matrixVectorMult(F, this.state);

    // Predict covariance: P = F * P * F^T + Q
    const FP = this.matrixMult(F, this.covariance);
    const FPFT = this.matrixMult(FP, this.transpose(F));
    const Q = this.createIdentityMatrix(6).map((row) =>
      row.map((val) => val * this.processNoise)
    );
    this.covariance = this.matrixAdd(FPFT, Q);
  }

  /**
   * Update step: incorporate sensor measurement
   */
  update(measurement: Vector3D, measurementNoise: number): void {
    // Measurement matrix (observe position only)
    const H = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
    ];

    // Measurement noise covariance
    const R = this.createIdentityMatrix(3).map((row) =>
      row.map((val) => val * measurementNoise)
    );

    // Innovation: y = z - H * x
    const Hx = this.matrixVectorMult(H, this.state);
    const z = [measurement.x, measurement.y, measurement.z];
    const y = z.map((val, i) => val - Hx[i]);

    // Innovation covariance: S = H * P * H^T + R
    const HP = this.matrixMult(H, this.covariance);
    const HPHT = this.matrixMult(HP, this.transpose(H));
    const S = this.matrixAdd(HPHT, R);

    // Kalman gain: K = P * H^T * S^-1
    const HT = this.transpose(H);
    const PHT = this.matrixMult(this.covariance, HT);
    const Sinv = this.matrixInverse(S);
    const K = this.matrixMult(PHT, Sinv);

    // Update state: x = x + K * y
    const Ky = this.matrixVectorMult(K, y);
    this.state = this.state.map((val, i) => val + Ky[i]);

    // Update covariance: P = (I - K * H) * P
    const KH = this.matrixMult(K, H);
    const I = this.createIdentityMatrix(6);
    const IKH = this.matrixSubtract(I, KH);
    this.covariance = this.matrixMult(IKH, this.covariance);
  }

  /**
   * Fuse multiple sensor readings
   */
  fuseReadings(readings: SensorReading[]): FusedState {
    // Sort by timestamp
    readings.sort((a, b) => a.timestamp - b.timestamp);

    for (const reading of readings) {
      // Predict forward to reading time
      this.predict(reading.latency);

      // Update with reading
      const noise = this.measurementNoise / reading.accuracy;
      this.update(reading.value, noise);
    }

    return {
      position: {
        x: this.state[0],
        y: this.state[1],
        z: this.state[2],
      },
      velocity: {
        x: this.state[3],
        y: this.state[4],
        z: this.state[5],
      },
      covariance: cloneDeep(this.covariance),
      confidence: 1.0 / (1.0 + this.trace(this.covariance)),
      timestamp: Date.now(),
    };
  }

  // Matrix operations
  private createIdentityMatrix(n: number): number[][] {
    return Array(n)
      .fill(0)
      .map((_, i) =>
        Array(n)
          .fill(0)
          .map((_, j) => (i === j ? 1 : 0))
      );
  }

  private matrixMult(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const result: number[][] = Array(rows)
      .fill(0)
      .map(() => Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        for (let k = 0; k < inner; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  }

  private matrixVectorMult(A: number[][], v: number[]): number[] {
    return A.map((row) => row.reduce((sum, val, i) => sum + val * v[i], 0));
  }

  private matrixAdd(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  }

  private matrixSubtract(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  }

  private transpose(A: number[][]): number[][] {
    return A[0].map((_, i) => A.map((row) => row[i]));
  }

  private matrixInverse(A: number[][]): number[][] {
    const n = A.length;
    const aug = A.map((row, i) => [...row, ...this.createIdentityMatrix(n)[i]]);

    // Gaussian elimination (simplified)
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
      }
      [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];

      for (let k = i + 1; k < n; k++) {
        const factor = aug[k][i] / (aug[i][i] + 1e-10);
        for (let j = i; j < 2 * n; j++) {
          aug[k][j] -= factor * aug[i][j];
        }
      }
    }

    for (let i = n - 1; i >= 0; i--) {
      for (let k = i - 1; k >= 0; k--) {
        const factor = aug[k][i] / (aug[i][i] + 1e-10);
        for (let j = 0; j < 2 * n; j++) {
          aug[k][j] -= factor * aug[i][j];
        }
      }
      const divisor = aug[i][i] + 1e-10;
      for (let j = 0; j < 2 * n; j++) {
        aug[i][j] /= divisor;
      }
    }

    return aug.map((row) => row.slice(n));
  }

  private trace(A: number[][]): number {
    return A.reduce((sum, row, i) => sum + row[i], 0);
  }

  getState(): { position: Vector3D; velocity: Vector3D } {
    return {
      position: { x: this.state[0], y: this.state[1], z: this.state[2] },
      velocity: { x: this.state[3], y: this.state[4], z: this.state[5] },
    };
  }
}

// ============================================================================
// Reality Bridge (Bidirectional Sync)
// ============================================================================

class RealityBridge {
  private physicalStates: Map<EntityId, PhysicalState>;
  private digitalStates: Map<EntityId, DigitalState>;
  private sensorFusions: Map<EntityId, SensorFusion>;
  private syncMetrics: Map<EntityId, SyncMetrics>;

  constructor() {
    this.physicalStates = new Map();
    this.digitalStates = new Map();
    this.sensorFusions = new Map();
    this.syncMetrics = new Map();
  }

  /**
   * Register entity for tracking
   */
  registerEntity(entityId: EntityId, initialPosition: Vector3D): void {
    this.physicalStates.set(entityId, {
      entityId,
      timestamp: Date.now(),
      position: initialPosition,
      velocity: { x: 0, y: 0, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      properties: new Map(),
    });

    this.digitalStates.set(entityId, {
      entityId,
      timestamp: Date.now(),
      position: initialPosition,
      velocity: { x: 0, y: 0, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      predictedPosition: initialPosition,
      confidence: 1.0,
    });

    this.sensorFusions.set(entityId, new SensorFusion(initialPosition));

    this.syncMetrics.set(entityId, {
      latency: 0,
      drift: 0,
      confidence: 1.0,
      lastSync: Date.now(),
    });
  }

  /**
   * Update from physical sensors
   */
  updateFromPhysical(entityId: EntityId, readings: SensorReading[]): void {
    const fusion = this.sensorFusions.get(entityId);
    if (!fusion) return;

    const fusedState = fusion.fuseReadings(readings);
    const now = Date.now();

    // Update physical state
    this.physicalStates.set(entityId, {
      entityId,
      timestamp: now,
      position: fusedState.position,
      velocity: fusedState.velocity,
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      properties: new Map(),
    });

    // Update sync metrics
    const digitalState = this.digitalStates.get(entityId)!;
    const drift = this.computeDrift(fusedState.position, digitalState.position);

    this.syncMetrics.set(entityId, {
      latency: now - fusedState.timestamp,
      drift,
      confidence: fusedState.confidence,
      lastSync: now,
    });
  }

  /**
   * Update from digital simulation
   */
  updateFromDigital(entityId: EntityId, newState: DigitalState): void {
    this.digitalStates.set(entityId, newState);
  }

  /**
   * Sync digital to physical (command actuation)
   */
  syncDigitalToPhysical(entityId: EntityId): PhysicalState | null {
    const digitalState = this.digitalStates.get(entityId);
    if (!digitalState) return null;

    // In real system, this would send commands to actuators
    const commandedState: PhysicalState = {
      entityId,
      timestamp: Date.now(),
      position: digitalState.position,
      velocity: digitalState.velocity,
      orientation: digitalState.orientation,
      properties: new Map(),
    };

    return commandedState;
  }

  /**
   * Get synchronization status
   */
  getSyncStatus(entityId: EntityId): SyncMetrics | null {
    return this.syncMetrics.get(entityId) ?? null;
  }

  /**
   * Get all states
   */
  getStates(entityId: EntityId): {
    physical: PhysicalState | undefined;
    digital: DigitalState | undefined;
  } {
    return {
      physical: this.physicalStates.get(entityId),
      digital: this.digitalStates.get(entityId),
    };
  }

  private computeDrift(pos1: Vector3D, pos2: Vector3D): number {
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    const dz = pos1.z - pos2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

// ============================================================================
// Predictive Sync Engine
// ============================================================================

class PredictiveSyncEngine {
  private bridge: RealityBridge;
  private predictionHorizon: number;
  private compensationEnabled: boolean;

  constructor(bridge: RealityBridge, predictionHorizon: number = 100) {
    this.bridge = bridge;
    this.predictionHorizon = predictionHorizon; // milliseconds
    this.compensationEnabled = true;
  }

  /**
   * Predict future state with latency compensation
   */
  predictState(entityId: EntityId, targetTime: Timestamp): DigitalState | null {
    const states = this.bridge.getStates(entityId);
    const physicalState = states.physical;
    if (!physicalState) return null;

    const dt = (targetTime - physicalState.timestamp) / 1000; // to seconds

    // Dead reckoning: predict based on current velocity
    const predictedPosition: Vector3D = {
      x: physicalState.position.x + physicalState.velocity.x * dt,
      y: physicalState.position.y + physicalState.velocity.y * dt,
      z: physicalState.position.z + physicalState.velocity.z * dt,
    };

    // Compute prediction confidence (decays with time)
    const confidence = Math.exp(-dt / 1.0);

    return {
      entityId,
      timestamp: targetTime,
      position: physicalState.position,
      velocity: physicalState.velocity,
      orientation: physicalState.orientation,
      predictedPosition,
      confidence,
    };
  }

  /**
   * Synchronize with adaptive compensation
   */
  async synchronize(entityId: EntityId): Promise<void> {
    const syncMetrics = this.bridge.getSyncStatus(entityId);
    if (!syncMetrics) return;

    // If drift is high, apply correction
    if (syncMetrics.drift > 0.1) {
      const now = Date.now();
      const predictedState = this.predictState(
        entityId,
        now + this.predictionHorizon
      );

      if (predictedState && this.compensationEnabled) {
        // Apply predictive compensation
        this.bridge.updateFromDigital(entityId, predictedState);
      }
    }
  }

  /**
   * Enable/disable latency compensation
   */
  setCompensation(enabled: boolean): void {
    this.compensationEnabled = enabled;
  }

  /**
   * Set prediction horizon
   */
  setPredictionHorizon(milliseconds: number): void {
    this.predictionHorizon = milliseconds;
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateHybridReality(): Promise<void> {
  console.log("=".repeat(80));
  console.log("HYBRID REALITY TWINS DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Sensor Fusion
  console.log("\n📡 Demo 1: Multi-Sensor Fusion with Kalman Filter");
  console.log("-".repeat(80));

  const fusion = new SensorFusion(
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
    0.05,
    0.3
  );

  console.log("Initial state: position=(0,0,0), velocity=(1,0,0)");

  // Simulate noisy sensor readings
  const readings: SensorReading[] = [
    {
      sensorId: "gps",
      timestamp: 100,
      value: { x: 0.15, y: 0.02, z: -0.01 },
      accuracy: 0.8,
      latency: 50,
    },
    {
      sensorId: "imu",
      timestamp: 100,
      value: { x: 0.12, y: -0.01, z: 0.01 },
      accuracy: 0.95,
      latency: 10,
    },
    {
      sensorId: "lidar",
      timestamp: 100,
      value: { x: 0.14, y: 0.0, z: 0.0 },
      accuracy: 0.99,
      latency: 20,
    },
  ];

  console.log("\nSensor readings (with noise):");
  for (const r of readings) {
    console.log(
      `  ${r.sensorId}: (${r.value.x.toFixed(3)}, ${r.value.y.toFixed(3)}, ${r.value.z.toFixed(3)}) ` +
        `acc=${r.accuracy}, latency=${r.latency}ms`
    );
  }

  const fusedState = fusion.fuseReadings(readings);
  console.log(
    `\nFused estimate: (${fusedState.position.x.toFixed(3)}, ${fusedState.position.y.toFixed(3)}, ${fusedState.position.z.toFixed(3)})`
  );
  console.log(`Confidence: ${(fusedState.confidence * 100).toFixed(1)}%`);

  // Demo 2: Reality Bridge
  console.log("\n🌉 Demo 2: Bidirectional Reality Bridge");
  console.log("-".repeat(80));

  const bridge = new RealityBridge();
  bridge.registerEntity("drone-1", { x: 0, y: 0, z: 10 });

  console.log("Registered entity: drone-1 at (0, 0, 10)");

  // Simulate physical sensor updates
  const droneReadings: SensorReading[] = [
    {
      sensorId: "gps",
      timestamp: Date.now(),
      value: { x: 1.2, y: 0.5, z: 10.1 },
      accuracy: 0.85,
      latency: 40,
    },
  ];

  bridge.updateFromPhysical("drone-1", droneReadings);

  const states = bridge.getStates("drone-1");
  console.log(
    `\nPhysical state: (${states.physical?.position.x.toFixed(2)}, ${states.physical?.position.y.toFixed(2)}, ${states.physical?.position.z.toFixed(2)})`
  );
  console.log(
    `Digital state: (${states.digital?.position.x.toFixed(2)}, ${states.digital?.position.y.toFixed(2)}, ${states.digital?.position.z.toFixed(2)})`
  );

  const syncStatus = bridge.getSyncStatus("drone-1");
  console.log(`\nSync metrics:`);
  console.log(`  Latency: ${syncStatus?.latency.toFixed(0)}ms`);
  console.log(`  Drift: ${syncStatus?.drift.toFixed(3)}m`);
  console.log(
    `  Confidence: ${((syncStatus?.confidence ?? 0) * 100).toFixed(1)}%`
  );

  // Demo 3: Predictive Synchronization
  console.log("\n🔮 Demo 3: Predictive Synchronization");
  console.log("-".repeat(80));

  const predictor = new PredictiveSyncEngine(bridge, 100);

  console.log("Prediction horizon: 100ms");

  const futureTime = Date.now() + 100;
  const predictedState = predictor.predictState("drone-1", futureTime);

  console.log(
    `\nCurrent position: (${states.physical?.position.x.toFixed(2)}, ${states.physical?.position.y.toFixed(2)}, ${states.physical?.position.z.toFixed(2)})`
  );
  console.log(
    `Predicted position (+100ms): (${predictedState?.predictedPosition.x.toFixed(2)}, ${predictedState?.predictedPosition.y.toFixed(2)}, ${predictedState?.predictedPosition.z.toFixed(2)})`
  );
  console.log(
    `Prediction confidence: ${((predictedState?.confidence ?? 0) * 100).toFixed(1)}%`
  );

  // Demo 4: Adaptive Compensation
  console.log("\n⚙️ Demo 4: Adaptive Latency Compensation");
  console.log("-".repeat(80));

  console.log("Enabling latency compensation...");
  predictor.setCompensation(true);

  await predictor.synchronize("drone-1");
  console.log("Synchronization complete with compensation");

  console.log("\nDisabling latency compensation...");
  predictor.setCompensation(false);
  await predictor.synchronize("drone-1");
  console.log("Synchronization complete without compensation");

  // Demo 5: Real-time Tracking Simulation
  console.log("\n📊 Demo 5: Real-Time Tracking Simulation");
  console.log("-".repeat(80));

  console.log("Simulating 5 time steps with moving drone...\n");

  for (let t = 0; t < 5; t++) {
    const time = Date.now();
    const trueX = t * 0.5;
    const trueY = t * 0.2;
    const trueZ = 10 + t * 0.1;

    // Add sensor noise
    const noisyReadings: SensorReading[] = [
      {
        sensorId: "gps",
        timestamp: time,
        value: {
          x: trueX + (Math.random() - 0.5) * 0.2,
          y: trueY + (Math.random() - 0.5) * 0.2,
          z: trueZ + (Math.random() - 0.5) * 0.2,
        },
        accuracy: 0.8,
        latency: 50,
      },
    ];

    bridge.updateFromPhysical("drone-1", noisyReadings);
    const currentStates = bridge.getStates("drone-1");
    const metrics = bridge.getSyncStatus("drone-1");

    console.log(`Step ${t + 1}:`);
    console.log(
      `  True: (${trueX.toFixed(2)}, ${trueY.toFixed(2)}, ${trueZ.toFixed(2)})`
    );
    console.log(
      `  Estimated: (${currentStates.physical?.position.x.toFixed(2)}, ${currentStates.physical?.position.y.toFixed(2)}, ${currentStates.physical?.position.z.toFixed(2)})`
    );
    console.log(`  Drift: ${metrics?.drift.toFixed(3)}m\n`);
  }

  console.log("✅ Hybrid Reality Twins demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  SensorFusion,
  RealityBridge,
  PredictiveSyncEngine,
  type Vector3D,
  type Quaternion,
  type PhysicalState,
  type DigitalState,
  type SensorReading,
  type FusedState,
  type SyncMetrics,
};
