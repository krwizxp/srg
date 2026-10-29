from __future__ import annotations

import argparse
from decimal import Decimal as D, getcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import mpmath as mp
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

SOURCE_DIRECTORY = Path(__file__).resolve().parent
getcontext().prec = 80
mp.mp.dps = 70
ALPHA = D("0.01")
TEST_COUNT = 4
MASK = MultiPolygon([
    Polygon([(0, 0), (0.75, 0), (0.75, 0.75), (0, 0.75)],
            holes=[[(0.25, 0.25), (0.5, 0.25), (0.5, 0.5), (0.25, 0.5)]]),
    box(0.875, 0.75, 1, 1),
    box(0.875, 0.125, 0.9375, 0.1875),
])
assert MASK.is_valid and MASK.bounds == (0.0, 0.0, 1.0, 1.0)
shapely.prepare(MASK)
XCUTS = list(map(D, ["0", "0.25", "0.5", "0.75", "0.875", "0.9375", "1"]))
YCUTS = list(map(D, ["0", "0.125", "0.1875", "0.25", "0.5", "0.75", "1"]))
MODULUS = 2**64


def save(output_directory, name, data):
    (output_directory / name).write_text(json.dumps(data, indent=2) + "\n")


def classify_decimal(x, y):
    main = (D(0) <= x <= D("0.75") and D(0) <= y <= D("0.75")
            and not (D("0.25") < x < D("0.5") and D("0.25") < y < D("0.5")))
    island_a = D("0.875") <= x <= D(1) and D("0.75") <= y <= D(1)
    island_b = D("0.875") <= x <= D("0.9375") and D("0.125") <= y <= D("0.1875")
    assert sum((main, island_a, island_b)) <= 1
    return 1 if main else 2 if island_a else 3 if island_b else 0


def classify_geos(points):
    answer = np.zeros(len(points), dtype=np.uint8)
    geometries = shapely.points(points)
    for index, component in enumerate(MASK.geoms, 1):
        selected = shapely.covers(component, geometries)
        assert not np.any(answer[selected])
        answer[selected] = index
    assert np.array_equal(answer != 0, shapely.covers(MASK, geometries))
    return answer


def axis(bits, cuts):
    count = 2**bits
    decimals = [(D(index) + D("0.5")) / count for index in range(count)]
    floats = np.array([float(value) for value in decimals])
    step = D(1) / count
    assert decimals[0] > 0 and decimals[-1] < 1
    assert all(b - a == step for a, b in zip(decimals[:-1], decimals[1:]))
    assert all(D.from_float(value) == exact for value, exact in zip(floats, decimals))
    assert len(np.unique(floats)) == count
    assert not any(value in cuts for value in decimals)
    classes = np.searchsorted(np.array(list(map(float, cuts))), floats, side="right") - 1
    assert np.all(classes >= 0) and np.all(classes < len(cuts) - 1)
    return decimals, floats, classes


def sum_range(first, last):
    return (first + last) * (last - first + 1) // 2


def sum_squares(first, last):
    def prefix(n):
        return n * (n + 1) * (2 * n + 1) // 6
    return prefix(last) - prefix(first - 1)


def xor_range(first, last):
    def prefix(n):
        return 0 if n < 0 else [n, 1, n + 1, 0][n % 4]
    return prefix(last) ^ prefix(first - 1)


def factorized_oracle(bits):
    n = 2**bits
    xd, xf, xclass = axis(bits, XCUTS)
    yd, yf, yclass = axis(bits, YCUTS)
    for polygon in MASK.geoms:
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                assert x0 == x1 or y0 == y1
                assert D.from_float(x0) in XCUTS and D.from_float(y0) in YCUTS
                assert D.from_float(x1) in XCUTS and D.from_float(y1) in YCUTS
    block_categories = np.empty((len(YCUTS) - 1, len(XCUTS) - 1), dtype=np.uint8)
    component_counts = [0, 0, 0]
    accepted_sum = accepted_square_sum = accepted_xor = 0
    oracle_points = 0
    boundaries = []
    for yi, (ylo, yhi) in enumerate(zip(YCUTS[:-1], YCUTS[1:])):
        representative_y = (ylo + yhi) / 2
        points = np.column_stack([xf, np.full(n, float(representative_y))])
        actual = classify_geos(points)
        expected = np.array([classify_decimal(x, representative_y) for x in xd], dtype=np.uint8)
        assert np.array_equal(actual, expected)
        oracle_points += n
        ys = np.flatnonzero(yclass == yi)
        yfirst, ylast = int(ys[0]), int(ys[-1])
        assert len(ys) == ylast - yfirst + 1
        for xi, (xlo, xhi) in enumerate(zip(XCUTS[:-1], XCUTS[1:])):
            xs = np.flatnonzero(xclass == xi)
            xfirst, xlast = int(xs[0]), int(xs[-1])
            assert len(xs) == xlast - xfirst + 1
            category = classify_decimal((xlo + xhi) / 2, representative_y)
            assert np.all(actual[xs] == category)
            block_categories[yi, xi] = category
            if not category:
                continue
            nx, ny = len(xs), len(ys)
            component_counts[category - 1] += nx * ny
            sx, sy = sum_range(xfirst, xlast), sum_range(yfirst, ylast)
            accepted_sum += nx * n * sy + ny * sx
            accepted_square_sum += (nx * n*n * sum_squares(yfirst, ylast)
                                    + 2*n*sx*sy + ny * sum_squares(xfirst, xlast))
            accepted_xor ^= ((xor_range(yfirst, ylast) * n if nx % 2 else 0)
                             ^ (xor_range(xfirst, xlast) if ny % 2 else 0))
    for yi in range(n):
        for cut in XCUTS[1:-1]:
            for xi in [int(cut*n) - 1, int(cut*n)]:
                boundaries.append((xi, yi))
    for xi in range(n):
        for cut in YCUTS[1:-1]:
            for yi in [int(cut*n) - 1, int(cut*n)]:
                boundaries.append((xi, yi))
    boundary_indices = np.array(boundaries, dtype=np.int64)
    boundary_points = np.column_stack([xf[boundary_indices[:, 0]], yf[boundary_indices[:, 1]]])
    boundary_actual = classify_geos(boundary_points)
    boundary_expected = block_categories[yclass[boundary_indices[:, 1]], xclass[boundary_indices[:, 0]]]
    assert np.array_equal(boundary_actual, boundary_expected)
    accepted = sum(component_counts)
    assert Fraction(accepted, n*n) == Fraction(137, 256)
    result = {"axis_bits": bits, "word_bits": bits * 2, "candidate_count": n*n,
              "accepted_candidates": accepted, "rejected_candidates": n*n - accepted,
              "component_counts": component_counts,
              "accepted_word_sum_mod_2_64": str(accepted_sum % MODULUS),
              "accepted_word_square_sum_mod_2_64": str(accepted_square_sum % MODULUS),
              "accepted_word_xor": str(accepted_xor),
              "axis_centres_checked_per_axis": n, "axis_step_decimal": str(D(1)/n),
              "axis_centres_exact_in_binary64": True, "axis_centres_unique": True,
              "cartesian_partition_cells": int(block_categories.size),
              "geos_and_decimal_representative_row_points": oracle_points,
              "geos_boundary_adjacent_grid_points": len(boundaries),
              "geos_decimal_mismatches": 0,
              "factorization_proof": "Every polygon edge lies on a declared horizontal/vertical cut. Membership is constant in each open Cartesian cell. Every axis centre lies in exactly one cell; multiply independently enumerated x/y centre counts."}
    return result, (xd, xf, xclass, yd, yf, yclass, block_categories)


def full_python_oracle(bits, oracle):
    xd, xf, xclass, yd, yf, yclass, block = oracle
    n = 2**bits
    points = np.column_stack([np.tile(xf, n), np.repeat(yf, n)])
    geos = classify_geos(points)
    decimal = np.array([classify_decimal(x, y) for y in yd for x in xd], dtype=np.uint8)
    factorized = block[np.repeat(yclass, n), np.tile(xclass, n)]
    assert np.array_equal(geos, decimal) and np.array_equal(geos, factorized)
    return geos


def enumerate_two_words(categories):
    c = len(categories)
    assert c == 256
    accepted = categories != 0
    outcomes = np.full(c*c, -1, dtype=np.int64)
    initial = np.repeat(np.arange(c), c)
    supplemental = np.tile(np.arange(c), c)
    initial_inside = accepted[initial]
    final_inside = ~initial_inside & accepted[supplemental]
    outcomes[initial_inside] = initial[initial_inside]
    outcomes[final_inside] = supplemental[final_inside]
    initial_counts = np.bincount(outcomes[initial_inside], minlength=c)
    retry_counts = np.bincount(outcomes[final_inside], minlength=c)
    total_counts = np.bincount(outcomes[outcomes >= 0], minlength=c)
    m, r = int(accepted.sum()), int((~accepted).sum())
    assert np.all(initial_counts[accepted] == c)
    assert np.all(retry_counts[accepted] == r)
    assert np.all(total_counts[accepted] == c + r)
    assert np.all(total_counts[~accepted] == 0)
    failures = int(np.count_nonzero(outcomes < 0))
    assert failures == r*r and m*(c+r) + failures == c*c
    return {"axis_bits": 4, "word_bits": 8, "candidate_count": c,
            "accepted_candidates": m, "rejected_candidates": r,
            "two_word_paths_individually_enumerated": len(outcomes),
            "initial_acceptance_paths_per_valid_word": c,
            "one_rejection_then_acceptance_paths_per_valid_word": r,
            "successful_paths_per_valid_word": c + r,
            "distinct_successful_weights": np.unique(total_counts[accepted]).tolist(),
            "failed_paths": failures, "successful_paths": int(np.count_nonzero(outcomes >= 0)),
            "conditional_probability_per_valid_word": str(Fraction(1, m)),
            "accepted_word_histogram": {str(word): int(total_counts[word]) for word in range(c) if accepted[word]},
            "unused_second_word_note": "For initial acceptance, all C possible latent second words represent the same terminated outcome; the provider is not read. This makes every two-word path have the common weight 1/C²."}


def weighted_proof(c, m):
    r = c - m
    result = []
    for limit in [0, 1, 2, 3, 1024]:
        total_paths = c**(limit + 1)
        failure_paths = r**(limit + 1)
        weight = (total_paths - failure_paths) // m
        assert m*weight + failure_paths == total_paths
        if limit <= 3:
            expanded = sum(r**k * c**(limit-k) for k in range(limit + 1))
            assert expanded == weight
        probability = (D(r)/c)**(limit + 1)
        entry = {"supplemental_read_limit": limit,
                 "path_space_word_count": limit + 1,
                 "per_valid_word_path_weight_is_identical": True,
                 "exact_normalization_identity_verified": True,
                 "failure_probability_decimal": str(probability),
                 "conditional_probability_per_valid_word_exact": str(Fraction(1, m)),
                 "weight_bit_length": weight.bit_length(),
                 "weight_integer_sha256": hashlib.sha256(weight.to_bytes((weight.bit_length()+7)//8, "big")).hexdigest()}
        if limit <= 3:
            entry.update({"total_equiprobable_paths": str(total_paths),
                          "failure_paths": str(failure_paths), "successful_paths_per_valid_word": str(weight)})
        result.append(entry)
    return {"candidate_count": c, "accepted_count": m, "rejected_count": r,
            "first_acceptance_at_k_rejections_probability_per_valid_word": "R^k / C^(k+1)",
            "cap_L_equiprobable_path_weight_per_valid_word": "sum(k=0..L, R^k*C^(L-k)) = (C^(L+1)-R^(L+1))/M",
            "conditional_uniformity": "Divide the identical per-word success mass by M times that mass: P(output=j | success)=1/M for every accepted j.",
            "unbounded_rejection": "Sum(k>=0, R^k/C^(k+1))=1/(C-R)=1/M when 0<M<=C.",
            "limits": result,
            "path_enumeration_note": "These large path-space counts are exact integer algebra, not individually enumerated paths. Only the 8-bit-total-word two-word model is path-enumerated separately."}


def chi_square(counts, weights):
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    expected = counts.sum() * weights / weights.sum()
    assert expected.min() >= 5
    statistic = float(np.sum((counts - expected)**2 / expected))
    df = len(counts) - 1
    probability = mp.gammainc(mp.mpf(df)/2, mp.mpf(statistic)/2, mp.inf)/mp.gamma(mp.mpf(df)/2)
    threshold = mp.mpf(str(ALPHA))/TEST_COUNT
    return {"observed_counts": counts.astype(int).tolist(), "expected_counts": expected.tolist(),
            "chi_square": statistic, "degrees_of_freedom": df, "p_value": str(probability),
            "predeclared_bonferroni_threshold": str(threshold), "passes": probability >= threshold}


def rejection_simulation(bits, seed, oracle, categories=None):
    _, _, xclass, _, _, yclass, blocks = oracle
    n = 2**bits
    rng = random.Random(seed)
    output_words = []
    component_counts = [0, 0, 0]
    reads = []
    for _ in range(100_000):
        supplementary = 0
        word = rng.getrandbits(bits*2)
        while True:
            category = int(blocks[yclass[word >> bits], xclass[word & (n-1)]])
            if category:
                component_counts[category-1] += 1
                output_words.append(word)
                reads.append(supplementary)
                break
            assert supplementary < 1024
            word = rng.getrandbits(bits*2)
            supplementary += 1
    result = {"axis_bits": bits, "word_bits": bits*2, "seed": seed, "accepted_output_count": len(output_words),
              "component_counts": component_counts,
              "component_frequency_test": chi_square(component_counts, [128, 8, 1]),
              "supplemental_reads": {"total": sum(reads), "mean": sum(reads)/len(reads),
                  "expected_mean_exact": str(Fraction(119, 137)), "maximum": max(reads)},
              "source": "Deterministic Python random.Random.getrandbits test stream; no reuse of rejected words/axes"}
    if bits == 4:
        accepted = categories != 0
        counts = np.bincount(output_words, minlength=n*n)
        assert np.all(counts[~accepted] == 0)
        result["every_accepted_grid_point_frequency_test"] = chi_square(counts[accepted], np.ones(int(accepted.sum())))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path,
                        default=Path('build/reduced-lattice-validation'))
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    executable = output_directory / 'reduced_lattice_exhaustive'
    compile_command = [
        'cc', '-O3', '-march=native', '-std=c11', '-Wall', '-Wextra', '-Werror',
        str(SOURCE_DIRECTORY / 'reduced_lattice_exhaustive.c'), '-o', str(executable),
    ]
    total_started = time.perf_counter()
    subprocess.run(compile_command, check=True)
    compile_seconds = time.perf_counter() - total_started
    c_outputs = {
        4: 'reduced-lattice-word8-exhaustive.json',
        8: 'reduced-lattice-axis8-exhaustive.json',
        16: 'reduced-lattice-axis16-exhaustive.json',
    }
    c_started = time.perf_counter()
    for bits, name in c_outputs.items():
        result = subprocess.run([str(executable), str(bits)], check=True,
                                capture_output=True, text=True)
        save(output_directory, name, json.loads(result.stdout))
    c_execution_seconds = time.perf_counter() - c_started
    started = time.perf_counter()
    models, proofs, statistics = [], [], []
    path_proof = None
    for model_index, bits in enumerate([4, 8, 16]):
        oracle_result, oracle = factorized_oracle(bits)
        c = json.loads((output_directory / c_outputs[bits]).read_text())
        for key in ["accepted_candidates", "rejected_candidates", "component_counts",
                    "accepted_word_sum_mod_2_64", "accepted_word_square_sum_mod_2_64", "accepted_word_xor"]:
            assert c[key] == oracle_result[key], (bits, key, c[key], oracle_result[key])
        assert c["visited_candidates"] == oracle_result["candidate_count"]
        categories = full_python_oracle(bits, oracle) if bits <= 8 else None
        if bits == 4:
            path_proof = enumerate_two_words(categories)
        models.append({"independent_decimal_geos_oracle": oracle_result, "c_exhaustive_enumeration": c,
                       "python_geos_decimal_full_2d_points_enumerated": 2**(2*bits) if bits <= 8 else 0,
                       "c_and_independent_oracle_counts_and_checksums_agree": True})
        proofs.append(weighted_proof(c["visited_candidates"], c["accepted_candidates"]))
        statistics.append(rejection_simulation(bits, 20261029 + model_index, oracle, categories))
        print(f"Verified axis_bits={bits}: candidates={c['visited_candidates']}, accepted={c['accepted_candidates']}", flush=True)
    save(output_directory, "reduced-lattice-two-word-paths.json", path_proof)
    report = {
        "status": "all_exact_checks_passed",
        "specification_interpretation": "The required 8-bit and 16-bit axes mean 2^16 and 2^32 candidate words. Both were exhaustively iterated by the independent C program. A supplementary 8-bit total-word (4+4-axis) model also exhausts all two-word paths.",
        "mask": {"coordinate_system": "normalized equal-area rectangle [0,1]², axes longitude and Q(latitude)",
                 "mainland": "[0,3/4]² minus open hole (1/4,1/2)²",
                 "island_a": "[7/8,1] x [3/4,1]", "island_b": "[7/8,15/16] x [1/8,3/16]",
                 "continuous_area_exact": "137/256", "component_areas_exact": ["1/2", "1/32", "1/256"],
                 "closed_boundary_policy": "covers; outer and hole boundaries included; hole interior excluded",
                 "all_edges_axis_aligned": True,
                 "geographic_preimage_note": "Axis-aligned normalized-Q edges become straight constant-latitude or constant-longitude geographic edges; no real polygon edges are remapped as new straight segments."},
        "models": models, "weighted_conditional_uniformity": proofs,
        "predeclared_familywise_alpha": str(ALPHA), "predeclared_statistical_test_count": TEST_COUNT,
        "reduced_rejection_statistics": statistics,
        "actual_64_bit_candidate_space_exhaustively_enumerated": False,
        "limits": ["The real 32+32-bit candidate space has 2^64 points and was not enumerated.",
                   "The exact accepted-grid count of the real Korea mask was not computed from its continuous area.",
                   "Uniformity requires independent uniform initial and supplementary words; arbitrary manual inputs have no such statistical guarantee.",
                   "Statistical p-values are regression checks. Exact equal weighting follows from enumeration and the independent algebraic proof.",
                   "The independent C program is a synthetic oracle, not a claim that every production code path was exercised by 2^32 native conversions."],
        "compiler": subprocess.run(["cc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
        "compile_command": compile_command,
        "validation_command": [sys.executable, *sys.argv],
        "software": {"numpy": np.__version__, "shapely": shapely.__version__, "GEOS": shapely.geos_version_string,
                     "decimal_precision": getcontext().prec, "mpmath_precision": mp.mp.dps},
        "source_sha256": {name: hashlib.sha256((SOURCE_DIRECTORY/name).read_bytes()).hexdigest()
                          for name in ["verify_reduced_lattice.py", "reduced_lattice_exhaustive.c"]},
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "compile_elapsed_seconds": compile_seconds,
        "c_execution_elapsed_seconds": c_execution_seconds,
        "python_verification_elapsed_seconds": time.perf_counter() - started,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    save(output_directory, "reduced-lattice-validation.json", report)
    concise = {
        "domain": "C=2^64 equally likely complete words; M accepted grid centres; R=C-M",
        "assumptions": ["Every candidate uses both axes from the same word", "Every supplementary word is independent and uniform", "Membership is deterministic for the immutable boundary version"],
        "per_accepted_word_first_success_after_k_rejections": "R^k/C^(k+1)",
        "with_supplemental_limit_L": "P(output=j and success)=(1-(R/C)^(L+1))/M",
        "conditional_result": "P(output=j | success)=1/M for every accepted j",
        "limit_failure_probability": "(R/C)^(L+1)", "production_supplemental_limit": 1024,
        "exact_exhaustive_reduced_candidate_counts": [256, 65536, 4294967296],
        "exact_exhaustive_two_word_path_count_for_8_bit_total_words": 65536,
        "actual_2_to_64_candidate_enumeration_claimed": False,
        "all_accepted_reduced_points_have_identical_weight": True,
        "continuous_uniformity_claimed": False,
        "evidence": ["reduced-lattice-validation.json", "reduced-lattice-two-word-paths.json"],
    }
    save(output_directory, "finite-lattice-proof.json", concise)
    print(json.dumps({"status": report["status"], "models": [m["c_exhaustive_enumeration"] for m in models],
                      "statistical_tests": [{"axis_bits": x["axis_bits"], "component_p": x["component_frequency_test"]["p_value"],
                           "full_point_p": x.get("every_accepted_grid_point_frequency_test", {}).get("p_value")}
                          for x in statistics], "seconds": report["python_verification_elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
