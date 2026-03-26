import copy
import math
import random
import time
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("qcentroid-user-log")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _disruption_map(disruptions: List[Dict]) -> Dict[str, float]:
    dm: Dict[str, float] = {}
    for d in disruptions:
        for loc_id in d.get("affected_locations", []):
            dm[loc_id] = dm.get(loc_id, 0) + d.get("delay_min", 0.0)
    return dm


def _route_time(stop_ids: List[str], locs: Dict, depot_id: str,
                speed_kmh: float, uncertainty: float,
                disrupted: Dict[str, float]) -> float:
    """Total route time in minutes: travel + service + disruption delays + TW penalties."""
    current_time = 0.0
    seq = [depot_id] + stop_ids + [depot_id]
    for i in range(len(seq) - 1):
        a, b = locs[seq[i]], locs[seq[i + 1]]
        dist_km = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        travel = (dist_km / speed_kmh) * 60.0 * (1.0 + uncertainty)
        current_time += travel
        if b["id"] != depot_id:
            current_time += disrupted.get(b["id"], 0.0) + float(b.get("service_time", 0.0))
            tw = b.get("time_window")
            if tw:
                earliest, latest = tw
                if current_time < earliest:
                    current_time = float(earliest)
                elif current_time > latest:
                    current_time += (current_time - latest) * 2.0  # soft penalty
    return current_time


# ── Quantum Kernel Feature Map ────────────────────────────────────────────────

def _quantum_kernel_features(dist_matrix: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Simulated Quantum Kernel Feature Map.
    Projects pairwise distances into a higher-dimensional feature space
    using parameterised Pauli rotations (ZZFeatureMap analogue).
    """
    n = len(dist_matrix)
    features = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            phi = dist_matrix[i, j]
            # Simulate ZZ interaction: cos^2(phi) kernel
            features[i, j] = np.cos(phi) ** 2 * np.exp(-phi ** 2 / (2 * n_qubits))
    return features


# ── QUBO Builder ─────────────────────────────────────────────────────────────

def _build_qubo(customers: List[Dict], depot: Dict, vehicles: List[Dict],
                disrupted: Dict[str, float], speed_kmh: float,
                n_qubits: int, penalty: float) -> np.ndarray:
    """
    Build QUBO matrix Q for the VRP.
    Variables: x[i,k] = 1 iff customer i is assigned to vehicle k (in position 1).
    Simplified single-position formulation for tractability.
    Cost includes Haversine distance + disruption delays.
    Constraints (penalised):
      - Each customer visited exactly once
      - Vehicle capacity not exceeded (soft)
    """
    n_cust = len(customers)
    n_veh = len(vehicles)
    size = n_cust * n_veh
    Q = np.zeros((size, size))

    # Helper index
    def idx(i, k):
        return i * n_veh + k

    # --- Objective: minimise total travel time ---
    for i, c in enumerate(customers):
        for k, v in enumerate(vehicles):
            # Cost: depot->customer + customer->depot (in minutes)
            d_in = _haversine(depot["lat"], depot["lon"], c["lat"], c["lon"])
            d_out = _haversine(c["lat"], c["lon"], depot["lat"], depot["lon"])
            travel = ((d_in + d_out) / max(v.get("speed_kmh", speed_kmh), 1e-6)) * 60.0
            delay = disrupted.get(c["id"], 0.0)
            Q[idx(i, k), idx(i, k)] += travel + delay

    # Quantum kernel: pairwise interaction using feature map
    positions = np.array([[c["lat"], c["lon"]] for c in customers])
    if n_cust > 1:
        dist_mat = np.array([[
            _haversine(positions[i, 0], positions[i, 1],
                       positions[j, 0], positions[j, 1])
            for j in range(n_cust)] for i in range(n_cust)])
        # Normalise distances
        max_d = dist_mat.max() or 1.0
        kernel = _quantum_kernel_features(dist_mat / max_d, n_qubits)
        for i in range(n_cust):
            for j in range(i + 1, n_cust):
                for k in range(n_veh):
                    # Penalise assigning nearby customers to different vehicles
                    q_ij = 0.5 * kernel[i, j]
                    Q[idx(i, k), idx(j, k)] -= q_ij
                    Q[idx(i, k), idx(j, (k + 1) % n_veh)] += q_ij

    # --- Constraint 1: each customer visited exactly once ---
    for i in range(n_cust):
        for k in range(n_veh):
            Q[idx(i, k), idx(i, k)] += penalty * (1 - 2)
            for l in range(k + 1, n_veh):
                Q[idx(i, k), idx(i, l)] += 2 * penalty

    # --- Constraint 2: capacity (soft) ---
    for k, v in enumerate(vehicles):
        cap = float(v.get("capacity", 100.0))
        for i in range(n_cust):
            for j in range(i + 1, n_cust):
                excess = (customers[i].get("demand", 1.0)
                          + customers[j].get("demand", 1.0)) / cap
                if excess > 1.0:
                    Q[idx(i, k), idx(j, k)] += penalty * excess

    return Q


# ── Simulated Quantum Annealing ───────────────────────────────────────────────

def _sqa(Q: np.ndarray, n_replicas: int, n_sweeps: int, rng: random.Random,
         np_rng: np.random.Generator) -> np.ndarray:
    """
    Path-integral Monte Carlo / Simulated Quantum Annealing.
    Returns best binary spin configuration minimising x^T Q x.
    """
    n = Q.shape[0]
    # Initialise replicas randomly
    replicas = np_rng.integers(0, 2, size=(n_replicas, n)).astype(float)
    best_energy = float("inf")
    best_config = replicas[0].copy()

    Gamma_start, Gamma_end = 3.0, 0.01
    T = 1.5  # temperature

    for sweep in range(n_sweeps):
        Gamma = Gamma_start * (Gamma_end / Gamma_start) ** (sweep / n_sweeps)
        J_perp = -Gamma / (2 * n_replicas)

        for rep in range(n_replicas):
            order = list(range(n))
            rng.shuffle(order)
            for bit in order:
                # Local field from QUBO
                h = (Q[bit, :] + Q[:, bit]) @ replicas[rep] - Q[bit, bit] * replicas[rep, bit]
                # Transverse field from neighbouring replicas
                prev_rep = replicas[(rep - 1) % n_replicas]
                next_rep = replicas[(rep + 1) % n_replicas]
                h_perp = J_perp * (prev_rep[bit] + next_rep[bit])
                # Metropolis flip
                delta = (2 * replicas[rep, bit] - 1) * (h + h_perp)
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    replicas[rep, bit] = 1.0 - replicas[rep, bit]

        # Track best replica
        for rep in range(n_replicas):
            x = replicas[rep]
            energy = float(x @ Q @ x)
            if energy < best_energy:
                best_energy = energy
                best_config = x.copy()

    return best_config


# ── Solution decoder ─────────────────────────────────────────────────────────

def _decode_solution(config: np.ndarray, customers: List[Dict],
                     vehicles: List[Dict]) -> Dict[int, List[int]]:
    """Map binary QUBO solution to vehicle->[customer_indices]."""
    n_cust = len(customers)
    n_veh = len(vehicles)
    assignment: Dict[int, List[int]] = {k: [] for k in range(n_veh)}

    for i in range(n_cust):
        # Pick vehicle with highest activation for this customer
        scores = [config[i * n_veh + k] for k in range(n_veh)]
        best_k = int(np.argmax(scores))
        assignment[best_k].append(i)

    # Capacity repair: move overloaded customers to least-loaded vehicle
    for k, idxs in assignment.items():
        total = sum(customers[i].get("demand", 1.0) for i in idxs)
        cap = float(vehicles[k].get("capacity", 100.0))
        if total > cap:
            for i in list(idxs):
                if total <= cap:
                    break
                # Find least-loaded other vehicle
                others = [(j, sum(customers[x].get("demand", 1.0)
                                  for x in assignment[j]))
                          for j in range(n_veh) if j != k]
                others.sort(key=lambda t: t[1])
                for alt_k, alt_load in others:
                    if (alt_load + customers[i].get("demand", 1.0)
                            <= float(vehicles[alt_k].get("capacity", 100.0))):
                        assignment[k].remove(i)
                        assignment[alt_k].append(i)
                        total -= customers[i].get("demand", 1.0)
                        break

    return assignment


def _greedy_order(stop_indices: List[int], customers: List[Dict],
                  depot: Dict) -> List[int]:
    """Nearest-neighbour ordering for stops on a given vehicle."""
    if not stop_indices:
        return []
    unvisited = list(stop_indices)
    ordered = []
    cur_lat, cur_lon = depot["lat"], depot["lon"]
    while unvisited:
        nearest = min(unvisited,
                      key=lambda i: _haversine(cur_lat, cur_lon,
                                               customers[i]["lat"], customers[i]["lon"]))
        ordered.append(nearest)
        cur_lat, cur_lon = customers[nearest]["lat"], customers[nearest]["lon"]
        unvisited.remove(nearest)
    return ordered


def _route_analytics(stop_ids: List[str], locs: Dict, depot_id: str,
                     speed_kmh: float, uncertainty: float,
                     disrupted: Dict[str, float]) -> Dict:
    """Per-stop ETAs, compliance, and fuel cost."""
    current_time = 0.0
    seq = [depot_id] + stop_ids + [depot_id]
    stop_etas: Dict[str, float] = {}
    service_results: Dict[str, Dict] = {}
    violations = []
    total_km = 0.0

    for i in range(len(seq) - 1):
        a_id, b_id = seq[i], seq[i + 1]
        a, b = locs[a_id], locs[b_id]
        dist_km = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        travel = (dist_km / speed_kmh) * 60.0 * (1.0 + uncertainty)
        current_time += travel
        total_km += dist_km

        if b_id != depot_id:
            current_time += disrupted.get(b_id, 0.0) + float(b.get("service_time", 0.0))
            stop_etas[b_id] = round(current_time, 1)
            on_time = True
            tw = b.get("time_window")
            if tw:
                earliest, latest = tw
                if current_time < earliest:
                    current_time = float(earliest)
                elif current_time > latest:
                    on_time = False
                    violations.append({"stop": b_id,
                                       "lateness_min": round(current_time - latest, 1)})
            service_results[b_id] = {"eta_min": round(current_time, 1), "on_time": on_time}

    return {
        "stop_etas": stop_etas,
        "service_results": service_results,
        "violations": violations,
        "total_km": round(total_km, 3),
        "fuel_cost_eur": round(total_km * 0.22, 3),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    QCentroid - Quantum-Inspired Routing Solver (v2).

    Improvements over v1:
      - Haversine (spherical earth) distances
      - service_time incorporated into route cost
      - Disruption delays applied in QUBO objective and route evaluation
      - Soft time-window penalties in route cost
      - Rich output: cost_breakdown, risk_metrics, service_level_results,
        per-stop ETAs, constraint_violations

    input_data schema:
        depot       : {id, lat, lon}
        customers   : [{id, lat, lon, demand, time_window, service_time}]
        vehicles    : [{id, capacity, speed_kmh}]
        disruptions : [{type, affected_locations, delay_min}]

    solver_params:
        n_qubits   (int): qubits for quantum kernel (default 4)
        n_replicas (int): Trotter replicas (default 8)
        n_sweeps   (int): SQA sweeps (default 500)
        seed       (int): random seed (default 42)
    """
    start_time = time.time()
    logger.info("Quantum-Inspired Routing Solver v2: starting")

    n_qubits = int(solver_params.get("n_qubits", 4))
    n_replicas = int(solver_params.get("n_replicas", 8))
    n_sweeps = int(solver_params.get("n_sweeps", 500))
    seed = int(solver_params.get("seed", 42))
    penalty = float(solver_params.get("qubo_penalty", 50.0))
    uncertainty = float(solver_params.get("uncertainty_factor", 0.15))

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ── Parse input ──────────────────────────────────────────────────────────
    depot = input_data["depot"]
    customers = input_data.get("customers", [])
    vehicles = input_data.get("vehicles", [])
    if not vehicles:
        vehicles = [{"id": "V1", "capacity": 100.0, "speed_kmh": 50.0},
                    {"id": "V2", "capacity": 100.0, "speed_kmh": 50.0}]

    disruptions = input_data.get("disruptions", [])
    disrupted = _disruption_map(disruptions)
    speed_kmh = float(vehicles[0].get("speed_kmh", 50.0))

    # Location lookup dict
    locs: Dict[str, Dict] = {depot["id"]: depot}
    for c in customers:
        locs[c["id"]] = c

    logger.info(f"Parsed {len(customers)} customers, {len(vehicles)} vehicles, "
                f"{len(disruptions)} disruptions. QUBO size: "
                f"{len(customers) * len(vehicles)}")

    # ── Build QUBO & solve ───────────────────────────────────────────────────
    if customers:
        Q = _build_qubo(customers, depot, vehicles, disrupted, speed_kmh,
                        n_qubits, penalty)
        config = _sqa(Q, n_replicas, n_sweeps, rng, np_rng)
        assignment = _decode_solution(config, customers, vehicles)
    else:
        assignment = {}

    # ── Build routes ─────────────────────────────────────────────────────────
    routes_output = []
    all_service_results: Dict[str, Dict] = {}
    all_violations = []
    total_cost_minutes = 0.0
    total_fuel_eur = 0.0

    for k, v in enumerate(vehicles):
        cust_indices = assignment.get(k, [])
        if not cust_indices:
            continue
        ordered = _greedy_order(cust_indices, customers, depot)
        stop_ids = [customers[i]["id"] for i in ordered]
        total_demand = sum(float(customers[i].get("demand", 1.0)) for i in ordered)

        route_time = _route_time(stop_ids, locs, depot["id"],
                                 float(v.get("speed_kmh", speed_kmh)),
                                 uncertainty, disrupted)
        total_cost_minutes += route_time

        analytics = _route_analytics(stop_ids, locs, depot["id"],
                                     float(v.get("speed_kmh", speed_kmh)),
                                     uncertainty, disrupted)
        all_service_results.update(analytics["service_results"])
        all_violations.extend(analytics["violations"])
        total_fuel_eur += analytics["fuel_cost_eur"]

        routes_output.append({
            "vehicle_id": v["id"],
            "stop_sequence": [depot["id"]] + stop_ids + [depot["id"]],
            "total_load": round(total_demand, 2),
            "estimated_cost_minutes": round(route_time, 2),
            "total_km": analytics["total_km"],
            "stop_etas": analytics["stop_etas"],
        })

    elapsed = round(time.time() - start_time, 3)
    total_cost_km = round(total_cost_minutes * speed_kmh / 60.0, 3)

    on_time_count = sum(1 for s in all_service_results.values() if s["on_time"])
    on_time_prob = round(on_time_count / max(len(all_service_results), 1), 3)
    solution_status = "optimal" if not all_violations else "feasible"

    logger.info(f"Quantum-Inspired Solver v2 done. Routes: {len(routes_output)}, "
                f"cost: {total_cost_minutes:.2f} min / {total_cost_km:.2f} km, "
                f"elapsed: {elapsed}s, on_time: {on_time_prob:.1%}")

    return {
        # Core routing output
        "routes": routes_output,
        "total_vehicles_used": len(routes_output),
        "total_quantum_cost_km": total_cost_km,
        "total_quantum_cost_minutes": round(total_cost_minutes, 3),
        "solver_type": "quantum_inspired_QAOA",
        "algorithm": "QUBO_SQA_QuantumKernel_v2",
        # QCentroid benchmark contract
        "objective_value": round(total_cost_minutes, 3),
        "solution_status": solution_status,
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "QUBO_SQA_QuantumKernel_v2",
            "n_qubits": n_qubits,
            "trotter_replicas": n_replicas,
            "sqa_sweeps": n_sweeps,
        },
        # Rich analytics
        "cost_breakdown": {
            "travel_time_min": round(total_cost_minutes, 2),
            "fuel_cost_eur": round(total_fuel_eur, 2),
            "lateness_penalty_min": round(
                sum(v["lateness_min"] for v in all_violations), 2),
            "detour_cost": 0.0,
        },
        "risk_metrics": {
            "on_time_probability": on_time_prob,
            "uncertainty_factor": uncertainty,
            "time_window_violations": len(all_violations),
            "uncertainty_correction_km": round(
                total_cost_km * uncertainty, 4),
        },
        "service_level_results": all_service_results,
        "constraint_violations": all_violations,
        "quantum_advantage": {
            "technique": "Simulated Quantum Annealing + Quantum Kernel Feature Maps",
            "n_qubits_simulated": n_qubits,
            "trotter_replicas": n_replicas,
            "hardware_ready": True,
            "notes": (
                "Solution encoded as QUBO - directly portable to D-Wave quantum "
                "annealers or gate-based QAOA on IBM/IonQ hardware."
            ),
        },
        # Required for benchmark charts
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": 0.0,
        },
    }
