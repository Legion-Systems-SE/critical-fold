"""
Tension Differential Engine — structural encoding of real numbers.

Encodes any real number as a digit sequence, computes first and second
differences (velocity/acceleration), discrete Laplacian at center,
pairwise dot products, cross-Laplacians, and collapse chains.

Usage:
    python manifold_sim/tension.py                    # full demo
    python manifold_sim/tension.py --pair c h         # pairwise analysis
    python manifold_sim/tension.py --matrix           # all-pairs comparison
    python manifold_sim/tension.py --collapse c h     # collapse chain test
    python manifold_sim/tension.py --sweep            # all-pairs collapse sweep
    python manifold_sim/tension.py --bridge           # orbit-tension bridge scan
    python manifold_sim/tension.py --bridge 182 273   # trace specific values

Authors: Mattias Hammarsten (framework), Claude/Anthropic (implementation)
Date: 2026-04-25, Uppsala
"""

from __future__ import annotations
import sys
import math
from fractions import Fraction
from typing import Optional

# =====================================================================
# CONSTANTS REGISTRY
# =====================================================================

CONSTANTS = {
    "c":     (299792458,          "speed of light (m/s, exact)"),
    "h":     ("6.62607015",       "Planck constant (×10⁻³⁴ J·s, exact 2019 SI)"),
    "k":     ("1.380649",         "Boltzmann constant (×10⁻²³ J/K, exact 2019 SI)"),
    "Na":    ("6.02214076",       "Avogadro number (×10²³, exact 2019 SI)"),
    "e":     ("1.602176634",      "elementary charge (×10⁻¹⁹ C, exact 2019 SI)"),
    "G":     ("6.67430",          "gravitational constant (×10⁻¹¹, measured)"),
    "alpha": ("7.2973525693",     "fine structure constant (×10⁻³, CODATA 2018)"),
    "pi":    ("3.14159265358979", "π"),
    "euler": ("2.71828182845904", "e (Euler's number)"),
    "phi":   ("1.61803398874989", "φ (golden ratio)"),
    "z1":    ("14.134725141734",  "first zeta zero (imaginary part)"),
    "z2":    ("21.022039638771",  "second zeta zero"),
    "z3":    ("25.010857580145",  "third zeta zero"),
    "P":     (3485,               "8-prime weft sum"),
    "omega": (3677,               "vector length (prime)"),
    "rho":   (3163,               "primitive root (C-P) mod Ω"),
    "fc":    (192,                "forbidden center"),
}

# Engine structural constants for cross-referencing
STRUCTURAL = {
    14: "√196 (baseline root)",
    13: "π(41) = shadow number",
    24: "|O| (survivor count)",
    127: "M₇ (7th Mersenne prime)",
    192: "forbidden center = Ω−P",
    196: "14² (baseline count)",
    182: "13×14",
    41: "smallest odd core",
    43: "core (172)",
    47: "core (188)",
    49: "7² core (196)",
    51: "core (204)",
    61: "largest odd core",
    919: "(Ω−1)/4",
    85: "P/41 = 5×17",
    -9: "center of Δ²(c)",
    -13: "∇²(z1 center) = cross-∇²(c,h)",
}


# =====================================================================
# PRIMITIVE ROOT ORBIT
# =====================================================================

RHO = 3163
OMEGA = 3677
N_ORB = OMEGA - 1

_orbit_cache = {}


def orbit():
    """Generate the primitive root orbit ρ^n mod Ω (cached)."""
    if "seq" not in _orbit_cache:
        seq = []
        val = 1
        for n in range(N_ORB):
            seq.append(val)
            val = (val * RHO) % OMEGA
        dlog = {v: n for n, v in enumerate(seq)}
        _orbit_cache["seq"] = seq
        _orbit_cache["dlog"] = dlog
    return _orbit_cache["seq"], _orbit_cache["dlog"]


def orbit_at(n: int) -> int:
    """ρ^n mod Ω."""
    seq, _ = orbit()
    return seq[n % N_ORB]


def orbit_addr(v: int) -> Optional[int]:
    """Discrete log: find n such that ρ^n ≡ v (mod Ω)."""
    _, dlog = orbit()
    return dlog.get(v % OMEGA)


def structural_tag(v: int) -> Optional[str]:
    """Check if a value is a known structural constant."""
    return STRUCTURAL.get(v)


# =====================================================================
# ORBIT-TENSION BRIDGE
# =====================================================================

def tension_of(v: int) -> Optional[int]:
    """Δ² of a 3-digit integer, collapsed to single value. None if not 3 digits."""
    if abs(v) < 100 or abs(v) >= 1000:
        return None
    d = [int(x) for x in str(abs(v))]
    return d[0] - 2 * d[1] + d[2]


def bridge_trace(value: int, depth: int = 6) -> list[dict]:
    """Follow a value through both the orbit and tension frameworks.

    At each step:
      - Look up the value in the orbit (address and neighbors)
      - Compute its tension differential
      - Check if results are structural constants
      - Feed results forward

    Returns a chain of steps.
    """
    chain = []
    seen = set()

    for step in range(depth):
        if value in seen:
            chain.append({"step": step, "value": value, "note": "CYCLE"})
            break
        seen.add(value)

        entry = {"step": step, "value": value, "tags": []}

        # structural tag
        tag = structural_tag(value)
        if tag:
            entry["tags"].append(f"structural: {tag}")
        tag_neg = structural_tag(-value)
        if tag_neg:
            entry["tags"].append(f"structural(-): {tag_neg}")

        # orbit address
        addr = orbit_addr(value) if 0 < value < OMEGA else None
        if addr is not None:
            entry["orbit_addr"] = addr
            addr_tag = structural_tag(addr)
            if addr_tag:
                entry["tags"].append(f"addr={addr}: {addr_tag}")

        # tension differential (3-digit values)
        t = tension_of(value)
        if t is not None:
            entry["tension"] = t
            t_tag = structural_tag(t)
            if t_tag:
                entry["tags"].append(f"Δ²={t}: {t_tag}")

        # prime counting function
        try:
            from sympy import primepi
            if 2 <= abs(value) <= 10000:
                pv = int(primepi(abs(value)))
                entry["prime_count"] = pv
                pc_tag = structural_tag(pv)
                if pc_tag:
                    entry["tags"].append(f"π({abs(value)})={pv}: {pc_tag}")
        except ImportError:
            pass

        # ρ mod value (if small enough to be meaningful)
        if 2 <= abs(value) <= 200:
            rm = RHO % abs(value)
            entry["rho_mod"] = rm
            rm_tag = structural_tag(rm)
            if rm_tag:
                entry["tags"].append(f"ρ mod {abs(value)}={rm}: {rm_tag}")

        chain.append(entry)

        # decide next value: prefer tension if available, else π
        if t is not None:
            value = abs(t)
        elif "prime_count" in entry:
            value = entry["prime_count"]
        else:
            break

    return chain


def print_bridge_trace(start_value: int, depth: int = 6):
    """Print a formatted bridge trace."""
    chain = bridge_trace(start_value, depth)
    print(f"\n  Bridge trace from {start_value}:")
    for entry in chain:
        s = entry["step"]
        v = entry["value"]
        tags = entry.get("tags", [])

        parts = [f"[{s}] value = {v}"]
        if "orbit_addr" in entry:
            parts.append(f"addr={entry['orbit_addr']}")
        if "tension" in entry:
            parts.append(f"Δ²={entry['tension']}")
        if "prime_count" in entry:
            parts.append(f"π={entry['prime_count']}")
        if "rho_mod" in entry:
            parts.append(f"ρ%{abs(v)}={entry['rho_mod']}")

        line = "    " + "  |  ".join(parts)
        print(line)
        for t in tags:
            print(f"      → {t}")

        if entry.get("note") == "CYCLE":
            print(f"      → CYCLE DETECTED")


def bridge_scan():
    """Scan all pairwise tension dot products and trace their connections."""
    from sympy import primepi

    L = 9
    keys = ["c", "h", "z1", "pi", "Na", "e", "alpha", "k"]

    profiles_map = {}
    for key in keys:
        val, desc = CONSTANTS[key]
        profiles_map[key] = profile(key, val, L)

    print("=" * 70)
    print("ORBIT-TENSION BRIDGE SCAN")
    print("=" * 70)

    print("\n  Centers:")
    for key in keys:
        c = profiles_map[key]["center"]
        tag = structural_tag(c) if c is not None else ""
        tag_str = f"  ← {tag}" if tag else ""
        print(f"    {key:>8}: center = {c}{tag_str}")

    print(f"\n  Dot products → orbit/structural cross-reference:")
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = dot(profiles_map[ka]["delta2"], profiles_map[kb]["delta2"])
            if d is None:
                continue

            parts = [f"{ka}·{kb} = {d}"]

            # tension of |dot|
            t = tension_of(abs(d))
            if t is not None:
                t_tag = structural_tag(t) or ""
                parts.append(f"Δ²(|{d}|)={t}")
                if t_tag:
                    parts.append(t_tag)

            # orbit address of |dot|
            if 0 < abs(d) < OMEGA:
                addr = orbit_addr(abs(d))
                if addr is not None:
                    a_tag = structural_tag(addr) or ""
                    parts.append(f"orbit addr={addr}")
                    if a_tag:
                        parts.append(a_tag)

            # structural tag of dot
            s_tag = structural_tag(d) or structural_tag(abs(d))
            if s_tag:
                parts.append(f"*** {s_tag} ***")

            # structural tag of -dot
            neg_tag = structural_tag(-d)
            if neg_tag and neg_tag != s_tag:
                parts.append(f"*** neg: {neg_tag} ***")

            print(f"    {'  |  '.join(parts)}")

    # trace the key dot products
    print(f"\n  --- Bridge traces for structurally tagged products ---")
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = dot(profiles_map[ka]["delta2"], profiles_map[kb]["delta2"])
            if d is not None and (structural_tag(d) or structural_tag(abs(d))
                                  or structural_tag(-d)
                                  or (tension_of(abs(d)) is not None
                                      and structural_tag(tension_of(abs(d))))):
                print_bridge_trace(abs(d))


# =====================================================================
# ENCODING
# =====================================================================

def encode(value, length: Optional[int] = None) -> list[int]:
    """Extract significant digits from a number.

    For integers: returns all digits.
    For reals (passed as string to preserve precision): strips the
    decimal point and returns significant digits.

    If length is specified, truncates or zero-pads to that length.
    """
    if isinstance(value, int):
        digits = [int(d) for d in str(abs(value))]
    elif isinstance(value, str):
        cleaned = value.lstrip("-").replace(".", "")
        # strip leading zeros only for numbers < 1 (like 0.00729...)
        if value.startswith("0.") or value.startswith("-0."):
            cleaned = cleaned.lstrip("0")
        digits = [int(d) for d in cleaned]
    elif isinstance(value, float):
        return encode(repr(value), length)
    else:
        raise TypeError(f"unsupported type: {type(value)}")

    if length is not None:
        digits = digits[:length]
        while len(digits) < length:
            digits.append(0)

    return digits


def encode_base(value: int, base: int) -> list[int]:
    """Convert a positive integer to a digit sequence in the given base."""
    if value == 0:
        return [0]
    digits = []
    n = abs(value)
    while n > 0:
        digits.append(n % base)
        n //= base
    digits.reverse()
    return digits


# =====================================================================
# DIFFERENCE OPERATORS
# =====================================================================

def delta1(seq: list[int]) -> list[int]:
    """First difference (velocity / polarity)."""
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]


def delta2(seq: list[int]) -> list[int]:
    """Second difference (acceleration / resolution)."""
    d1 = delta1(seq)
    return [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]


def delta2_from_value(value, length: Optional[int] = None) -> list[int]:
    """Convenience: value → digits → Δ²."""
    return delta2(encode(value, length))


# =====================================================================
# STRUCTURAL MEASURES
# =====================================================================

def amplitude(d1: list[int]) -> int:
    """Peak absolute amplitude of a first-difference sequence."""
    if not d1:
        return 0
    return max(abs(x) for x in d1)


def center_index(seq: list[int]) -> int:
    """Index of the central element."""
    return len(seq) // 2


def center_value(seq: list[int]) -> Optional[int]:
    """Value at the center of a sequence."""
    if not seq:
        return None
    return seq[center_index(seq)]


def laplacian_at(seq: list[int], i: int) -> Optional[int]:
    """Discrete 1D Laplacian: seq[i-1] - 2*seq[i] + seq[i+1]."""
    if i < 1 or i >= len(seq) - 1:
        return None
    return seq[i - 1] - 2 * seq[i] + seq[i + 1]


def laplacian_at_center(d2: list[int]) -> Optional[int]:
    """Discrete Laplacian at the center of a Δ² sequence."""
    if len(d2) < 3:
        return None
    return laplacian_at(d2, center_index(d2))


def zero_sum_pairs(seq: list[int]) -> list[tuple[int, int, int]]:
    """Find adjacent pairs that sum to zero. Returns (index, a, b)."""
    pairs = []
    for i in range(len(seq) - 1):
        if seq[i] + seq[i + 1] == 0:
            pairs.append((i, seq[i], seq[i + 1]))
    return pairs


# =====================================================================
# PAIRWISE OPERATIONS
# =====================================================================

def dot(d2_a: list[int], d2_b: list[int]) -> Optional[int]:
    """Dot product of two Δ² sequences. Requires equal length."""
    if len(d2_a) != len(d2_b):
        return None
    return sum(a * b for a, b in zip(d2_a, d2_b))


def cross_laplacian(d2_a: list[int], d2_b: list[int]) -> Optional[int]:
    """Cross-Laplacian: flanks of A around center of B.

    Uses the elements flanking A's center with B's center value:
    A[c-1] - 2*B[c] + A[c+1]
    where c = center index.
    """
    if len(d2_a) < 3 or len(d2_b) < 3:
        return None
    if len(d2_a) != len(d2_b):
        return None
    c = center_index(d2_a)
    if c < 1 or c >= len(d2_a) - 1:
        return None
    return d2_a[c - 1] - 2 * d2_b[c] + d2_a[c + 1]


# =====================================================================
# COLLAPSE CHAIN
# =====================================================================

def collapse(value, length: Optional[int] = None, max_depth: int = 10) -> list[dict]:
    """Recursive collapse: compute Δ² until it reduces to a single value.

    Returns the chain of states from the original through each reduction.
    """
    chain = []
    digits = encode(value, length)
    step = 0

    while len(digits) >= 3 and step < max_depth:
        d1 = delta1(digits)
        d2 = delta2(digits)
        chain.append({
            "step": step,
            "digits": digits,
            "length": len(digits),
            "delta1": d1,
            "delta2": d2,
            "center": center_value(d2) if d2 else None,
            "zero_sum_pairs": zero_sum_pairs(d2),
            "laplacian": laplacian_at_center(d2),
        })
        digits = [abs(x) for x in d2] if any(x < 0 for x in d2) else d2
        # for collapse, we work with absolute values of Δ² as new digits
        # but preserve the signed Δ² in the chain record
        digits = d2
        step += 1

        # if Δ² has only 1 or 2 elements, record final state and stop
        if len(digits) < 3:
            chain.append({
                "step": step,
                "digits": digits,
                "length": len(digits),
                "delta1": delta1(digits) if len(digits) >= 2 else [],
                "delta2": [],
                "center": digits[0] if len(digits) == 1 else None,
                "zero_sum_pairs": [],
                "laplacian": None,
            })
            break

    return chain


# =====================================================================
# FULL PROFILE
# =====================================================================

def profile(name: str, value, length: Optional[int] = None) -> dict:
    """Complete tension profile for a named constant."""
    digits = encode(value, length)
    d1 = delta1(digits)
    d2 = delta2(digits)

    return {
        "name": name,
        "value": value,
        "digits": digits,
        "length": len(digits),
        "delta1": d1,
        "delta2": d2,
        "amplitude": amplitude(d1) if d1 else 0,
        "center": center_value(d2) if d2 else None,
        "sum_d2": sum(d2) if d2 else 0,
        "abs_sum_d2": sum(abs(x) for x in d2) if d2 else 0,
        "zero_sum_pairs": zero_sum_pairs(d2),
        "laplacian": laplacian_at_center(d2),
    }


# =====================================================================
# DISPLAY
# =====================================================================

def print_profile(p: dict):
    print(f"\n  {p['name']} = {p['value']}")
    print(f"  digits:  {p['digits']}  (L={p['length']})")
    print(f"  Δ¹:     {p['delta1']}  A={p['amplitude']}")
    print(f"  Δ²:     {p['delta2']}")
    if p["delta2"]:
        print(f"  center:  {p['center']}  sum={p['sum_d2']}  |sum|={p['abs_sum_d2']}")
        if p["laplacian"] is not None:
            print(f"  ∇²(center): {p['laplacian']}")
        if p["zero_sum_pairs"]:
            for idx, a, b in p["zero_sum_pairs"]:
                print(f"  zero-sum pair at [{idx},{idx+1}]: ({a}, {b})")


def print_pair(name_a: str, p_a: dict, name_b: str, p_b: dict):
    d2a, d2b = p_a["delta2"], p_b["delta2"]
    d = dot(d2a, d2b)
    cl_ab = cross_laplacian(d2a, d2b)
    cl_ba = cross_laplacian(d2b, d2a)

    print(f"\n  {name_a} × {name_b}:")
    if d is not None:
        print(f"    dot(Δ²):  {d}")
    else:
        print(f"    dot(Δ²):  incompatible (L={len(d2a)} vs L={len(d2b)})")
    if cl_ab is not None:
        print(f"    cross-∇²: flanks({name_a})·center({name_b}) = {cl_ab}")
        print(f"    cross-∇²: flanks({name_b})·center({name_a}) = {cl_ba}")
        if cl_ab == cl_ba:
            print(f"    *** SYMMETRIC: both = {cl_ab} ***")


def print_collapse(name: str, value, length: Optional[int] = None):
    chain = collapse(value, length)
    print(f"\n  collapse({name} = {value}):")
    for state in chain:
        s = state["step"]
        d = state["digits"]
        c = state["center"]
        tag = f"  → center = {c}" if c is not None and len(d) <= 3 else ""
        if s == 0:
            print(f"    [{s}] digits {d} (L={state['length']})")
        else:
            print(f"    [{s}] Δ² = {d}{tag}")
    if chain:
        final = chain[-1]
        if final["center"] is not None:
            print(f"    terminal value: {final['center']}")
        elif len(final["digits"]) == 2:
            print(f"    terminal pair: {final['digits']}")


# =====================================================================
# CLI
# =====================================================================

def resolve(key: str, length: Optional[int] = None):
    """Resolve a constant name or literal to (name, value, length)."""
    if key in CONSTANTS:
        val, desc = CONSTANTS[key]
        return key, val, length
    try:
        return key, int(key), length
    except ValueError:
        pass
    try:
        return key, float(key), length
    except ValueError:
        pass
    return key, key, length


def main():
    args = sys.argv[1:]

    # detect --length N anywhere in args
    length = None
    if "--length" in args:
        idx = args.index("--length")
        length = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if not args or args[0] == "--demo":
        demo(length)
    elif args[0] == "--bridge":
        if len(args) > 1:
            for key in args[1:]:
                try:
                    v = int(key)
                    print_bridge_trace(v)
                except ValueError:
                    name, val, _ = resolve(key, length)
                    if isinstance(val, int):
                        print_bridge_trace(val)
        else:
            bridge_scan()
    elif args[0] == "--pair" and len(args) >= 3:
        pair_analysis(args[1], args[2], length)
    elif args[0] == "--matrix":
        matrix(args[1:] if len(args) > 1 else None, length)
    elif args[0] == "--collapse":
        for key in args[1:]:
            name, val, _ = resolve(key, length)
            print_collapse(name, val, length)
    elif args[0] == "--sweep":
        collapse_sweep(args[1:] if len(args) > 1 else None, length)
    elif args[0] == "--profile":
        for key in args[1:]:
            name, val, _ = resolve(key, length)
            p = profile(name, val, length)
            print_profile(p)
    else:
        for key in args:
            name, val, _ = resolve(key, length)
            p = profile(name, val, length)
            print_profile(p)


def demo(length: Optional[int] = None):
    print("=" * 70)
    print("TENSION DIFFERENTIAL ENGINE — DEMO")
    print("=" * 70)

    L = length or 9

    # group 1: constants with 9+ natural digits, shown at L
    group1_keys = ["c", "h", "z1", "pi", "Na", "e", "alpha"]
    # group 2: framework constants at natural length (no padding)
    group2_keys = ["P", "omega", "rho", "fc"]

    print(f"\n--- Profiles: physical constants (L={L}) ---")
    profiles = {}
    for key in group1_keys:
        val, desc = CONSTANTS[key]
        p = profile(key, val, L)
        profiles[key] = p
        print_profile(p)

    print(f"\n--- Profiles: framework constants (natural length) ---")
    for key in group2_keys:
        val, desc = CONSTANTS[key]
        p = profile(key, val)  # natural length, no padding
        profiles[key] = p
        print_profile(p)

    # pairwise within same-length groups
    g1 = {k: profiles[k] for k in group1_keys if profiles[k]["length"] == L}
    keys = list(g1.keys())

    if len(keys) >= 2:
        print(f"\n--- Pairwise (L={L} group) ---")
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                print_pair(ka, g1[ka], kb, g1[kb])

    # collapse chains
    print(f"\n--- Collapse chains ---")
    for key in ["c", "h", "z1"]:
        val, _ = CONSTANTS[key]
        print_collapse(key, val, L)

    # the -273 chain
    print(f"\n--- The c·h chain ---")
    d2c = profiles["c"]["delta2"]
    d2h = profiles["h"]["delta2"]
    if len(d2c) == len(d2h):
        d = dot(d2c, d2h)
        print(f"  dot(Δ²c, Δ²h) = {d}")
        if d is not None:
            print_collapse(f"|{d}|", abs(d))


def pair_analysis(key_a: str, key_b: str, length: Optional[int] = None):
    name_a, val_a, _ = resolve(key_a, length)
    name_b, val_b, _ = resolve(key_b, length)

    p_a = profile(name_a, val_a, length)
    p_b = profile(name_b, val_b, length)

    print("=" * 70)
    print(f"PAIR ANALYSIS: {name_a} × {name_b}")
    print("=" * 70)

    print_profile(p_a)
    print_profile(p_b)
    print_pair(name_a, p_a, name_b, p_b)

    d = dot(p_a["delta2"], p_b["delta2"])
    if d is not None and 100 <= abs(d) < 1000:
        print(f"\n  --- Collapse of dot product ---")
        print_collapse(f"|dot|={abs(d)}", abs(d))

    # check if dot product's collapse lands on either source's center
    if d is not None and 100 <= abs(d) < 1000:
        chain = collapse(abs(d))
        if chain and chain[-1]["center"] is not None:
            terminal = chain[-1]["center"]
            if terminal == p_a["center"]:
                print(f"\n  *** LOOP: collapse({abs(d)}) = {terminal}"
                      f" = center of {name_a} ***")
            if terminal == p_b["center"]:
                print(f"\n  *** LOOP: collapse({abs(d)}) = {terminal}"
                      f" = center of {name_b} ***")


def matrix(keys: Optional[list[str]] = None, length: Optional[int] = None):
    if keys is None or len(keys) == 0:
        keys = ["c", "h", "z1", "pi", "Na", "e", "k", "alpha"]

    L = length or 9
    print("=" * 70)
    print(f"COMPARISON MATRIX (L={L})")
    print("=" * 70)

    profiles_map = {}
    for key in keys:
        name, val, _ = resolve(key, L)
        profiles_map[key] = profile(name, val, L)
        print_profile(profiles_map[key])

    print(f"\n--- Dot products ---")
    header = f"  {'':>8}"
    for k in keys:
        header += f" {k:>8}"
    print(header)

    for ka in keys:
        row = f"  {ka:>8}"
        for kb in keys:
            d = dot(profiles_map[ka]["delta2"], profiles_map[kb]["delta2"])
            row += f" {d:>8}" if d is not None else f" {'—':>8}"
        print(row)

    print(f"\n--- Cross-Laplacians: flanks(row) · center(col) ---")
    print(header)
    for ka in keys:
        row = f"  {ka:>8}"
        for kb in keys:
            cl = cross_laplacian(
                profiles_map[ka]["delta2"],
                profiles_map[kb]["delta2"])
            row += f" {cl:>8}" if cl is not None else f" {'—':>8}"
        print(row)

    print(f"\n--- Collapse sweep (dot products) ---")
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = dot(profiles_map[ka]["delta2"], profiles_map[kb]["delta2"])
            if d is not None and abs(d) >= 10:
                chain = collapse(abs(d))
                terminal = chain[-1] if chain else None
                t_val = terminal["center"] if terminal and terminal["center"] is not None else \
                    terminal["digits"] if terminal else "?"
                # check for loops
                loop_target = None
                if terminal and terminal["center"] is not None:
                    for k in [ka, kb]:
                        if terminal["center"] == profiles_map[k]["center"]:
                            loop_target = k
                tag = f"  ← LOOP to {loop_target}" if loop_target else ""
                print(f"  {ka}·{kb} = {d:>6} → collapse to {t_val}{tag}")


def collapse_sweep(keys: Optional[list[str]] = None, length: Optional[int] = None):
    if keys is None or len(keys) == 0:
        keys = ["c", "h", "z1", "pi", "Na", "e", "k", "alpha"]

    L = length or 9
    print("=" * 70)
    print(f"COLLAPSE SWEEP (L={L})")
    print("=" * 70)

    profiles_map = {}
    centers = {}
    for key in keys:
        name, val, _ = resolve(key, L)
        p = profile(name, val, L)
        profiles_map[key] = p
        centers[key] = p["center"]
        print(f"  {key:>8}: center = {p['center']}")

    print(f"\n--- All pairs: dot → collapse → loop check ---")
    loops = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = dot(profiles_map[ka]["delta2"], profiles_map[kb]["delta2"])
            if d is None:
                continue
            chain = collapse(abs(d))
            terminal = chain[-1] if chain else None
            t_val = None
            if terminal:
                t_val = terminal["center"] if terminal["center"] is not None else \
                    terminal["digits"]

            loop_targets = []
            if terminal and terminal["center"] is not None:
                for k, cv in centers.items():
                    if terminal["center"] == cv:
                        loop_targets.append(k)

            tag = f"  ← loops to center of: {', '.join(loop_targets)}" \
                if loop_targets else ""
            print(f"  {ka:>8} · {kb:<8} = {d:>6}"
                  f"  collapse → {t_val}{tag}")
            if loop_targets:
                loops.append((ka, kb, d, t_val, loop_targets))

    if loops:
        print(f"\n--- LOOPS FOUND: {len(loops)} ---")
        for ka, kb, d, t, targets in loops:
            print(f"  {ka}·{kb} = {d} → {t} = center of {', '.join(targets)}")
    else:
        print(f"\n  No collapse loops detected at L={L}.")


if __name__ == "__main__":
    main()
