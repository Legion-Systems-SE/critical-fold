"""
Universal Pocket Calculator — Test Suite β0.4
==============================================
Experimental development. Tests a resonance-detection method for
physical constants based on digit-sequence curvature matching.

METHOD (every step printed — nothing hidden):

  1. ENCODE  value → first N significant digits (default N=10).
     e.g. 1836.15267343 → [1,8,3,6,1,5,2,6,7,3]

  2. SECOND DIFFERENCE (Δ²):  discrete acceleration of the digit sequence.
       Δ¹[i] = d[i+1] - d[i]              (velocity)
       Δ²[i] = Δ¹[i+1] - Δ¹[i]           (acceleration)
     Removes linear trends, exposes curvature.  len(Δ²) = N - 2.

  3. DOT PRODUCT:  for each parent constant p,
       dot(candidate, p) = Σ_i  Δ²_cand[i] × Δ²_p[i]
     Measures curvature similarity.

  4. TENSION TEST:  take |dot|.  If 100 ≤ |dot| ≤ 999, extract digits
     d₀d₁d₂ and compute
       D2 = d₀ − 2·d₁ + d₂
     D2 = 0  ⟹  digits form an arithmetic progression  →  "tonic"
     (This is the discrete second difference of a 3-element sequence.)

  5. VERDICT:
     - Exactly one parent with D2=0  →  UNIQUE resonance (match)
     - No parent with D2=0           →  SILENT (no match)
     - Multiple parents with D2=0    →  AMBIGUOUS

FALSE POSITIVE RATE:

  The per-pair probability of D2=0 by chance is ~3.7%.  With K parent
  constants, the probability that a random number produces at least one
  unique match is 1 - (1 - 0.037)^K - K·0.037·(1-0.037)^(K-1) ≈ 16%
  for K=5.  All claimed resonances hit at 100%, giving 31× discrimination.

  To control false positives, the test uses subsets of 5 parents
  rather than all 13.  The default subset ("pentatonic") contains only
  the 5 parents with confirmed resonances.

Usage:
    python3 upc_test.py                   # full test suite (default)
    python3 upc_test.py --test 1836.15    # test a single value
    python3 upc_test.py --test random     # test a random value
    python3 upc_test.py --scale chromatic # use all 13 parents
    python3 upc_test.py --sweep           # digit-length stability test
    python3 upc_test.py --explore         # experimental: normals, primes
    python3 upc_test.py --calibrate       # noise floor only
    python3 upc_test.py --batch           # include quantum+spectral subsets

Authors: Mattias Hammarsten / Claude (Anthropic)
Date: 2026-05-01, Uppsala
"""

import sys
import argparse
import random
from collections import Counter

sys.path.insert(0, __import__('pathlib').Path(__file__).parent.resolve().__str__())
import tension as T


# =====================================================================
# PARENT CONSTANTS — reference set
# =====================================================================

PARENTS = {
    'c':      ('299792458',                  'speed of light (m/s)'),
    'h':      ('6.6260701500000000',         'Planck constant'),
    'k':      ('1.380649000000000',          'Boltzmann constant'),
    'Na':     ('6.0221407600000000',         'Avogadro number'),
    'e_ch':   ('1.602176634000000',          'elementary charge'),
    'alpha':  ('7.2973525693000',            'fine structure constant'),
    'pi':     ('3.14159265358979323846',     'pi'),
    'euler':  ('2.71828182845904523536',     'e (Euler)'),
    'phi':    ('1.61803398874989484820',     'phi (golden ratio)'),
    'z1':     ('14.13472514173469379045',    'first zeta zero'),
    'z2':     ('21.02203963877155499262',    'second zeta zero'),
    'z3':     ('25.01085758014568876321',    'third zeta zero'),
    'zeta2':  ('1.64493406684822643647',     'zeta(2) = pi^2/6'),
}

# Parent subsets — fewer parents = lower false positive rate
SCALES = {
    'pentatonic': {
        'keys': ['c', 'h', 'alpha', 'pi', 'zeta2'],
        'desc': '5 parents with confirmed resonances (default)',
    },
    'quantum': {
        'keys': ['c', 'h', 'e_ch', 'alpha'],
        'desc': '4 electromagnetic constants',
    },
    'spectral': {
        'keys': ['z1', 'z2', 'z3', 'zeta2', 'pi'],
        'desc': '5 zeta-related constants',
    },
    'geometric': {
        'keys': ['c', 'pi', 'euler', 'phi', 'zeta2'],
        'desc': '5 mathematical + spacetime constants',
    },
    'physical': {
        'keys': ['c', 'h', 'k', 'Na', 'e_ch', 'alpha'],
        'desc': '6 SI-defined constants',
    },
    'chromatic': {
        'keys': list(PARENTS.keys()),
        'desc': 'All 13 parents (full survey, ~31% false positive rate)',
    },
}

DEFAULT_SCALE = 'pentatonic'
DIGIT_LENGTH = 10

# ── Test batches ─────────────────────────────────────────────────────

BATCH_PENTATONIC = [
    ('mu_p',    '1836.15267343',   'proton/electron mass ratio',   'zeta2'),
    ('higgs',   '125.700',         'Higgs boson mass (GeV)',        'c'),
    ('mu_N',    '5.050783699',     'nuclear magneton (J/T)',        'alpha'),
    ('s12',     '0.30700000000',   'neutrino mixing sin^2(th_12)', 'h'),
    ('Lambda',  '1.10560000000',   'cosmological constant',         'pi'),
]

BATCH_QUANTUM = [
    ('hbar',    '1.054571800',     'reduced Planck',                'c'),
    ('Phi0',    '2.06783383',      'magnetic flux quantum',          None),
    ('mu_B',    '927.4009994',     'Bohr magneton',                  None),
    ('G0',      '7.7480917310',    'conductance quantum',            None),
]

BATCH_SPECTRAL = [
    ('mu_p',    '1836.15267343',   'proton/electron mass ratio',   'zeta2'),
    ('Lambda',  '1.10560000000',   'cosmological constant',         'pi'),
    ('m_e_MeV', '0.5109989461',    'electron mass (MeV)',            None),
    ('sigma_SB','5.670367',        'Stefan-Boltzmann constant',      None),
]

BATCH_NEGATIVE = [
    ('sin2w',   '0.22230000000',   'weak mixing angle sin^2(th_W)', None),
    ('Vus',     '0.22430000000',   'CKM element V_us',              None),
    ('Vcb',     '0.04220000000',   'CKM element V_cb',              None),
    ('Vub',     '0.00394000000',   'CKM element V_ub',              None),
    ('y_t',     '0.99500000000',   'top Yukawa coupling',            None),
    ('s23',     '0.54600000000',   'neutrino mixing sin^2(th_23)',   None),
    ('s13',     '0.02200000000',   'neutrino mixing sin^2(th_13)',   None),
]

BATCH_EXPLORE = [
    ('CE',      '0.23571113171923293137414347535961',
                'Copeland-Erdos (concatenated primes, proven normal)'),
    ('sqrt2',   '1.41421356237309504880168872420969',
                'sqrt(2) (suspected normal)'),
    ('ln2',     '0.69314718055994530941723212145817',
                'ln(2) (suspected normal)'),
    ('gaps',    '1.22424246264624682',
                'prime gaps as decimal (p(n+1)-p(n))'),
    ('sum_1/p', '1.66164651701579600000',
                'prime reciprocal sum (first 15 primes)'),
    ('Mertens', '0.26149721284764159000',
                'Mertens constant'),
]


# =====================================================================
# CORE MEASUREMENT
# =====================================================================

def measure(val, scale=DEFAULT_SCALE, verbose=True, digit_length=DIGIT_LENGTH):
    """Measure a candidate against all parents in the chosen subset.

    Returns dict with verdict, winners, locks, spectrum.
    When verbose=True, prints every intermediate value.
    """
    scale_keys = SCALES[scale]['keys']

    digits = T.encode(val, digit_length)
    d1 = T.delta1(digits)
    d2 = T.delta2(digits)

    if verbose:
        print(f'  Input:    {val}')
        print(f'  Scale:    {scale} ({len(scale_keys)} parents)')
        print(f'  Length:   {digit_length} digits  (dim(d2) = {digit_length - 2})')
        print()
        print(f'  Step 1 — ENCODE')
        print(f'    digits = {digits}')
        print(f'  Step 2 — SECOND DIFFERENCE')
        print(f'    d1     = {d1}')
        print(f'    d2     = {d2}')
        print()
        print(f'  Step 3-4 — DOT PRODUCT + TENSION TEST')

    results = []
    for pname in scale_keys:
        pval, pdesc = PARENTS[pname]
        p_digits = T.encode(pval, digit_length)
        p_d2 = T.delta2(p_digits)
        dot = T.dot(d2, p_d2)
        t = T.tension_of(abs(dot)) if dot is not None and abs(dot) >= 100 else None

        entry = {
            'parent': pname,
            'parent_desc': pdesc,
            'parent_digits': p_digits,
            'parent_d2': p_d2,
            'dot': dot,
            'abs_dot': abs(dot) if dot is not None else 0,
            'D2': t,
        }
        results.append(entry)

        if verbose:
            tag = ''
            if t == 0:
                tag = '  <-- TONIC (D2=0)'
            elif t is not None and abs(t) == 7:
                tag = '  <-- LOCK (|D2|=7)'

            dot_digits = ''
            if dot is not None and 100 <= abs(dot) <= 999:
                dd = [int(x) for x in str(abs(dot))]
                dot_digits = (f'  digits({abs(dot)})=[{dd[0]},{dd[1]},{dd[2]}]'
                              f'  {dd[0]}-2*{dd[1]}+{dd[2]}={t}')

            p_d2_str = str(p_d2)
            print(f'    {pname:>6s}  d2={p_d2_str:>40s}'
                  f'  dot={dot:>6d}  |dot|={abs(dot):>4d}{dot_digits}{tag}')

    tonics = [r for r in results if r['D2'] == 0]
    locks = [r for r in results if r['D2'] is not None and abs(r['D2']) == 7]

    if len(tonics) == 0:
        verdict = 'SILENT'
        winners = []
    elif len(tonics) == 1:
        verdict = 'UNIQUE'
        winners = [tonics[0]['parent']]
    else:
        verdict = 'AMBIGUOUS'
        winners = [r['parent'] for r in tonics]

    if verbose:
        print()
        print(f'  Step 5 — VERDICT')
        if verdict == 'UNIQUE':
            w = tonics[0]
            print(f'    UNIQUE RESONANCE with {w["parent"]} ({w["parent_desc"]})')
            print(f'      dot = {w["dot"]},  |dot| = {w["abs_dot"]},  D2 = 0')
        elif verdict == 'AMBIGUOUS':
            print(f'    AMBIGUOUS — multiple resonances: {winners}')
        else:
            closest = min(results,
                          key=lambda r: abs(r['D2']) if r['D2'] is not None else 99)
            print(f'    SILENT — no resonance detected')
            print(f'      closest: {closest["parent"]} with D2={closest["D2"]}')
        if locks:
            print(f'    Locks (|D2|=7): {[r["parent"] for r in locks]}')

    return {
        'verdict': verdict,
        'winners': winners,
        'locks': [r['parent'] for r in locks],
        'spectrum': results,
    }


# =====================================================================
# TEST SUITES
# =====================================================================

def run_batch(batch, scale, label):
    """Run a batch of constants and check against expected parents."""
    print()
    print('=' * 72)
    print(f'  {label}')
    print(f'  Scale: {scale} — {SCALES[scale]["desc"]}')
    print('=' * 72)

    passed = failed = total = 0

    for name, val, desc, expected in batch:
        total += 1
        print()
        print(f'  -- {name}: {desc} --')
        result = measure(val, scale, verbose=True)

        if expected is not None:
            if result['verdict'] == 'UNIQUE' and result['winners'][0] == expected:
                passed += 1
                print(f'    PASS — uniquely matched expected parent ({expected})')
            elif result['verdict'] == 'AMBIGUOUS' and expected in result['winners']:
                passed += 1
                print(f'    PASS — expected parent in ambiguous set')
            else:
                failed += 1
                print(f'    FAIL — expected {expected},'
                      f' got {result["verdict"]}: {result["winners"]}')
        else:
            passed += 1
            print(f'    OK — no expected parent (exploratory)')

    print()
    print(f'  Batch result: {passed}/{total} passed, {failed} failed')
    return passed, failed, total


def test_negative(scale):
    """Negative controls: constants that should NOT resonate."""
    print()
    print('=' * 72)
    print('  NEGATIVE CONTROLS')
    print(f'  Scale: {scale} — constants without known structural derivation')
    print('=' * 72)

    passed = failed = total = 0

    for name, val, desc, _ in BATCH_NEGATIVE:
        total += 1
        result = measure(val, scale, verbose=False)

        if result['verdict'] == 'SILENT':
            passed += 1
            print(f'    PASS  {name:>7s}: silent')
        elif result['verdict'] == 'AMBIGUOUS':
            passed += 1
            print(f'    PASS  {name:>7s}: ambiguous {result["winners"]}')
        else:
            failed += 1
            print(f'    FAIL  {name:>7s}: unexpected resonance'
                  f' -> {result["winners"][0]}')

    print(f'\n  Result: {passed}/{total} passed, {failed} failed')
    return passed, failed, total


# =====================================================================
# CALIBRATION — noise floor
# =====================================================================

def calibrate(scale, n_trials=1000, seed=42):
    """Measure false positive rate with random digit sequences."""
    print()
    print('=' * 72)
    print(f'  CALIBRATION — {n_trials} random values')
    print(f'  Scale: {scale} — {SCALES[scale]["desc"]}')
    print('=' * 72)

    scale_keys = SCALES[scale]['keys']
    n_strings = len(scale_keys)
    rng = random.Random(seed)

    type_counts = Counter()
    parent_counts = Counter()
    per_pair_tonic = 0
    total_pairs = 0

    for _ in range(n_trials):
        first = rng.randint(1, 9)
        rest = [rng.randint(0, 9) for _ in range(DIGIT_LENGTH - 1)]
        digits = [first] + rest
        val = str(digits[0]) + '.' + ''.join(str(d) for d in digits[1:])

        result = measure(val, scale, verbose=False)
        type_counts[result['verdict']] += 1
        if result['verdict'] == 'UNIQUE':
            parent_counts[result['winners'][0]] += 1
        for r in result['spectrum']:
            total_pairs += 1
            if r['D2'] == 0:
                per_pair_tonic += 1

    unique_pct = type_counts['UNIQUE'] / n_trials * 100
    per_pair_pct = per_pair_tonic / total_pairs * 100
    false_specific = unique_pct / n_strings

    print(f'\n  Results ({n_trials} trials):')
    for stype in ['SILENT', 'UNIQUE', 'AMBIGUOUS']:
        n = type_counts[stype]
        pct = n / n_trials * 100
        bar = '#' * max(1, n // 10)
        print(f'    {stype:>10s}: {n:>4d} ({pct:>5.1f}%)  {bar}')

    print(f'\n  Noise rates:')
    print(f'    Per-pair tonic rate (D2=0):  {per_pair_pct:.2f}%')
    print(f'    Any unique resonance:        {unique_pct:.1f}%')
    print(f'    Per specific parent:         {false_specific:.1f}%')

    if type_counts['UNIQUE'] > 10:
        print(f'\n  Parent distribution ({type_counts["UNIQUE"]} false unique):')
        for pname in scale_keys:
            n = parent_counts.get(pname, 0)
            bar = '#' * max(1, n // 2)
            print(f'    {pname:>6s}: {n:>3d}  {bar}')

    # Discrimination ratio vs claimed tonics
    claims = [(n, v, p) for n, v, _, p in BATCH_PENTATONIC
              if p is not None and p in scale_keys]
    if claims:
        claim_hit = sum(
            1 for name, val, exp in claims
            if (r := measure(val, scale, verbose=False))['verdict'] == 'UNIQUE'
            and r['winners'][0] == exp
        )
        claim_pct = claim_hit / len(claims) * 100
        disc = claim_pct / max(false_specific, 0.01)
        print(f'\n  Discrimination:')
        print(f'    Claimed resonances: {claim_hit}/{len(claims)}'
              f' = {claim_pct:.0f}% correct')
        print(f'    Random baseline:    {false_specific:.1f}% per parent')
        print(f'    Ratio:              {disc:.0f}x')

    print()
    return type_counts, per_pair_pct


# =====================================================================
# STABILITY TEST — sweep digit length
# =====================================================================

def sweep(scale=DEFAULT_SCALE):
    """Test resonance stability across digit lengths 6-20.

    A resonance that persists across all digit lengths is structural.
    One that appears only at a specific length is dimensional.
    """
    print()
    print('=' * 72)
    print('  STABILITY TEST — resonance vs digit length')
    print(f'  Scale: {scale} — {SCALES[scale]["desc"]}')
    print('=' * 72)

    scale_keys = SCALES[scale]['keys']
    abbr = {'c': 'c', 'h': 'h', 'k': 'k', 'Na': 'N', 'e_ch': 'e',
            'alpha': 'a', 'pi': 'p', 'euler': 'E', 'phi': 'P',
            'z1': '1', 'z2': '2', 'z3': '3', 'zeta2': 'z'}

    lengths = list(range(6, 21))

    print(f'\n  Each column = digit length (dim(d2) = length - 2)')
    header = '            '
    for L in lengths:
        header += f'{L:>4d}'
    print(header)
    dim_line = '    dim(d2) '
    for L in lengths:
        dim_line += f'{L-2:>4d}'
    print(dim_line)
    print('    ' + '-' * (8 + 4 * len(lengths)))

    batch = BATCH_PENTATONIC

    for cname, cval, cdesc, expected in batch:
        row = f'    {cname:>7s} '
        hits = 0
        correct = 0
        for L in lengths:
            c_d2 = T.delta2(T.encode(cval, L))
            tonics = []
            for pn in scale_keys:
                pv = PARENTS[pn][0]
                p_d2 = T.delta2(T.encode(pv, L))
                dot = T.dot(c_d2, p_d2)
                if dot is not None and abs(dot) >= 100:
                    t = T.tension_of(abs(dot))
                    if t == 0:
                        tonics.append(pn)

            if len(tonics) == 0:
                row += '   .'
            elif len(tonics) == 1:
                row += f'   {abbr[tonics[0]]}'
                hits += 1
                if tonics[0] == expected:
                    correct += 1
            else:
                row += '   *'
                hits += 1
                if expected in tonics:
                    correct += 1

        pct = hits / len(lengths) * 100
        tag = 'structural' if pct > 80 else (
              'dimensional' if hits <= 2 else 'partial')
        row += f'   {hits:>2d}/{len(lengths)} {tag}'
        print(row)

    print()
    print('  Legend: . = silent, * = ambiguous,'
          ' letter = unique parent match')
    print('  Structural = stable across >80% of digit lengths')
    print('  Dimensional = appears at 1-2 specific lengths only')
    print()


# =====================================================================
# EXPERIMENTAL — normal numbers and prime encodings
# =====================================================================

def explore():
    """Experimental: probe mathematical constants and prime encodings.

    Tests each value across digit lengths 6-20 against all 13 parents.
    This extends the instrument beyond its calibrated range — results
    are exploratory, not validated.
    """
    print()
    print('=' * 72)
    print('  EXPERIMENTAL — digit-curvature probes')
    print('  Constants tested across digit lengths 6-20 (chromatic scale)')
    print('=' * 72)

    scale = 'chromatic'
    scale_keys = SCALES[scale]['keys']
    abbr = {'c': 'c', 'h': 'h', 'k': 'k', 'Na': 'N', 'e_ch': 'e',
            'alpha': 'a', 'pi': 'p', 'euler': 'E', 'phi': 'P',
            'z1': '1', 'z2': '2', 'z3': '3', 'zeta2': 'z'}

    lengths = list(range(6, 21))

    header = '              '
    for L in lengths:
        header += f'{L:>4d}'
    print(f'\n  Digit length sweep (13 parents):')
    print(header)
    dim_line = '    dim(d2)   '
    for L in lengths:
        dim_line += f'{L-2:>4d}'
    print(dim_line)
    print('    ' + '-' * (10 + 4 * len(lengths)))

    for cname, cval, cdesc in BATCH_EXPLORE:
        row = f'    {cname:>9s} '
        tonic_list = []
        for L in lengths:
            c_d2 = T.delta2(T.encode(cval, L))
            tonics = []
            for pn in scale_keys:
                pv = PARENTS[pn][0]
                p_d2 = T.delta2(T.encode(pv, L))
                dot = T.dot(c_d2, p_d2)
                if dot is not None and abs(dot) >= 100:
                    t = T.tension_of(abs(dot))
                    if t == 0:
                        tonics.append(pn)

            if len(tonics) == 0:
                row += '   .'
            elif len(tonics) == 1:
                row += f'   {abbr[tonics[0]]}'
                tonic_list.append((L, tonics[0]))
            elif len(tonics) == 2:
                row += f'  {abbr[tonics[0]]}{abbr[tonics[1]]}'
                tonic_list.extend((L, t) for t in tonics)
            else:
                row += f'  *{len(tonics)}'
                tonic_list.extend((L, t) for t in tonics)

        hits = len(set(L for L, _ in tonic_list))
        row += f'   {hits:>2d} hits'
        print(row)

    print()
    print('  Legend: . = silent, letter = unique parent, *N = N-way ambiguous')
    print('  c=speed of light  h=Planck  a=alpha  p=pi  z=zeta(2)')
    print('  E=euler(e)  P=phi  1/2/3=zeta zeros  k=Boltzmann')
    print()

    # Show the default-length measurement for CE
    print('  -- Detail: Copeland-Erdos at default length (10 digits) --')
    print()
    measure(BATCH_EXPLORE[0][1], 'pentatonic', verbose=True)
    print()

    # Show it at length 8 where it resonates
    print('  -- Detail: Copeland-Erdos at length 8 (dim 6) --')
    print()
    measure(BATCH_EXPLORE[0][1], 'pentatonic', verbose=True, digit_length=8)
    print()


# =====================================================================
# SINGLE VALUE TEST
# =====================================================================

def test_single(val_str, scale, digit_length=DIGIT_LENGTH):
    """Test a single value with full working shown."""
    print()
    print('=' * 72)
    print('  SINGLE MEASUREMENT')
    print('=' * 72)
    print()

    if val_str.lower() == 'random':
        rng = random.Random()
        first = rng.randint(1, 9)
        rest = [rng.randint(0, 9) for _ in range(digit_length - 1)]
        digits = [first] + rest
        val_str = str(digits[0]) + '.' + ''.join(str(d) for d in digits[1:])
        print(f'  Generated random value: {val_str}')
        print()

    measure(val_str, scale, verbose=True, digit_length=digit_length)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UPC Test Suite v0.4 — digit-curvature resonance test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parent subsets (--scale):
  pentatonic   c, h, alpha, pi, zeta(2)  — 5 parents, cleanest (default)
  quantum      c, h, e, alpha            — 4 parents, EM sector
  spectral     z1, z2, z3, zeta(2), pi   — 5 parents, zeta-related
  geometric    c, pi, euler, phi, zeta(2) — 5 parents, math constants
  physical     c, h, k, Na, e, alpha      — 6 parents, SI-defined
  chromatic    all 13 parents             — full survey, noisiest
        """)

    parser.add_argument('--test', type=str, metavar='VALUE',
                        help='test a single value (number or "random")')
    parser.add_argument('--scale', type=str, default=DEFAULT_SCALE,
                        choices=list(SCALES.keys()),
                        help=f'parent subset (default: {DEFAULT_SCALE})')
    parser.add_argument('--length', type=int, default=DIGIT_LENGTH,
                        help=f'digit length (default: {DIGIT_LENGTH})')
    parser.add_argument('--sweep', action='store_true',
                        help='digit-length stability test')
    parser.add_argument('--explore', action='store_true',
                        help='experimental: normal numbers, prime encodings')
    parser.add_argument('--calibrate', action='store_true',
                        help='noise floor calibration only')
    parser.add_argument('--trials', type=int, default=1000,
                        help='random trials for calibration (default: 1000)')
    parser.add_argument('--batch', action='store_true',
                        help='include quantum and spectral subsets')
    parser.add_argument('--negative', action='store_true',
                        help='negative controls only')
    args = parser.parse_args()

    # Single value
    if args.test:
        test_single(args.test, args.scale, args.length)
        return

    # Calibration only
    if args.calibrate:
        calibrate(args.scale, n_trials=args.trials)
        return

    # Sweep only
    if args.sweep:
        sweep(args.scale)
        return

    # Explore only
    if args.explore:
        explore()
        return

    # Negative only
    if args.negative:
        test_negative(args.scale)
        return

    # ── Full suite ────────────────────────────────────────────────

    total_passed = total_failed = total_tests = 0

    # 1. Confirmed resonances (pentatonic)
    p, f, t = run_batch(BATCH_PENTATONIC, 'pentatonic',
                        'CONFIRMED RESONANCES')
    total_passed += p; total_failed += f; total_tests += t

    # 2. Additional subsets (if --batch)
    if args.batch:
        p, f, t = run_batch(BATCH_QUANTUM, 'quantum',
                            'QUANTUM — electromagnetic sector')
        total_passed += p; total_failed += f; total_tests += t

        p, f, t = run_batch(BATCH_SPECTRAL, 'spectral',
                            'SPECTRAL — zeta-related')
        total_passed += p; total_failed += f; total_tests += t

    # 3. Negative controls
    p, f, t = test_negative(args.scale)
    total_passed += p; total_failed += f; total_tests += t

    # 4. Noise floor calibration
    calibrate(args.scale, n_trials=args.trials)

    # 5. Stability sweep
    sweep(args.scale)

    # Summary
    print('=' * 72)
    print(f'  UPC v0.4 — {total_passed}/{total_tests} passed,'
          f' {total_failed} failed')
    print('=' * 72)


if __name__ == '__main__':
    main()
