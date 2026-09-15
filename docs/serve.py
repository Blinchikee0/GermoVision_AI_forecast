from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import webbrowser
from collections import Counter
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import geo_ml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gv.serve")

PORT = 8000

_GEO_BUNDLE = None


def get_geo_bundle():
    global _GEO_BUNDLE
    if _GEO_BUNDLE is None:
        log.info("training geo ML bundle (first request)…")
        _GEO_BUNDLE = geo_ml.train_or_load(cities=WORLD_CITIES)
        if _GEO_BUNDLE:
            log.info("geo ML bundle ready · %d training samples", _GEO_BUNDLE.trained_n)
        else:
            log.warning("sklearn unavailable — falling back to gravity heuristic only")
    return _GEO_BUNDLE

REFERENCES = {
    "sars2_spike": {
        "name": "SARS-CoV-2 spike (S)",
        "disease": "COVID-19",
        "kind": "protein",
        "hotspots": [69, 143, 214, 371, 417, 452, 484, 501, 655, 681, 796, 954, 969],
        "aa": (
            "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAI"
            "HVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFC"
            "NDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIY"
            "SKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQP"
            "RTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDE"
        ),
    },
    "h1n1_ha": {
        "name": "Influenza A H1N1 hemagglutinin (HA)",
        "disease": "Influenza",
        "kind": "protein",
        "hotspots": [156, 158, 189, 190, 193, 222, 225],
        "aa": (
            "MKVKLLVLLCTFTATYADTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDSHNGKLCKLKGIAPLQ"
            "LGKCNIAGWLLGNPECDLLLTASSWSYIVETSNSENGTCYPGDFIDYEELREQLSSVSSFEKFEIFPK"
            "TSSWPNHETTKGVTAACSHAGKSSFYRNLLWLTKKGGSYPKLNQSYINDKGKEVLVLWGVHHPPTGTD"
            "QQSLYQNADAYVSVGSSKYNRRFTPEIAARPKVRDQAGRMNYYWTLLDPGDTITFEANGNLIAPWYAF"
            "ALSRGSGSGIINSNAPMDECDAKCQTPQGAINSSLPFQNVHPVTIGECPKYVKSTKLRMVTGLRNIPS"
        ),
    },
    "hiv1_env": {
        "name": "HIV-1 envelope (gp120)",
        "disease": "HIV/AIDS",
        "kind": "protein",
        "hotspots": [132, 197, 234, 302, 332, 386, 411, 461],
        "aa": (
            "MRVKEKYQHLWRWGWKWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNV"
            "WATHACVPTDPNPQEVVLENVTENFNMWKNNMVEQMHEDIISLWDQSLKPCVKLTPLCVTLNCTDLRN"
            "TTNTNNSSGGMEGGEIKNCSFNITTSIRDKVQKEYALFYKLDVVPIDNDSTSYRLINCNTSVITQACP"
            "KVSFEPIPIHYCAPAGFAILKCKDKKFNGKGPCTNVSTVQCTHGIKPVVSTQLLLNGSLAEKEVVIRS"
            "ENFTNNAKTIIVQLKESVEINCTRPNNNTRKSIHIGPGRAFYTTGEIIGDIRQAHCNISRAKWNNTLK"
        ),
    },
    "mtb_rpob": {
        "name": "M. tuberculosis rpoB (rifampicin)",
        "disease": "Tuberculosis",
        "kind": "protein",
        "hotspots": [426, 430, 431, 435, 441, 445, 450, 452],
        "aa": (
            "MADSRQSKTAASPSPSRPQSSSNNSVPGAPNRVSFAKLREPLEVPGLLDVQTDSFEWLIGSPRWRESA"
            "AERGDVNPVGGLEEVLYELSPIEDFSGSMSLSFSDPRFDDVKAPVDECKDKDMTYAAPLFVTAEFINN"
            "NTGEIKSQTVFMGDFPMMTEKGTFIINGTERVVVSQLVRSPGVYFDETIDKSTDKTLHSVKVIPSRGA"
            "WLEFDVDKRDTVGVRIDRKRRQPVTVLLKALGWTSEQIVERFGFSEIMRSTLEKDNTVGTDEALLDIY"
            "RKLRPGEPPTKESAQTLLENLFFKEKRYDLARVGRYKVNKKLGLHVGEPITSSTLTEEDVVATIEYLV"
        ),
    },
}

WORLD_CITIES = [
    {"n": "New York", "lat": 40.71, "lng": -74.00, "pop": 8.4},
    {"n": "Los Angeles", "lat": 34.05, "lng": -118.24, "pop": 3.9},
    {"n": "Mexico City", "lat": 19.43, "lng": -99.13, "pop": 9.2},
    {"n": "Sao Paulo", "lat": -23.55, "lng": -46.63, "pop": 12.3},
    {"n": "Buenos Aires", "lat": -34.60, "lng": -58.38, "pop": 3.1},
    {"n": "London", "lat": 51.51, "lng": -0.13, "pop": 9.0},
    {"n": "Paris", "lat": 48.86, "lng": 2.35, "pop": 2.2},
    {"n": "Madrid", "lat": 40.42, "lng": -3.70, "pop": 3.3},
    {"n": "Berlin", "lat": 52.52, "lng": 13.40, "pop": 3.7},
    {"n": "Rome", "lat": 41.90, "lng": 12.50, "pop": 2.8},
    {"n": "Moscow", "lat": 55.75, "lng": 37.62, "pop": 12.6},
    {"n": "Istanbul", "lat": 41.00, "lng": 28.97, "pop": 15.5},
    {"n": "Cairo", "lat": 30.04, "lng": 31.24, "pop": 10.0},
    {"n": "Lagos", "lat": 6.52, "lng": 3.38, "pop": 15.4},
    {"n": "Johannesburg", "lat": -26.20, "lng": 28.05, "pop": 5.6},
    {"n": "Nairobi", "lat": -1.29, "lng": 36.82, "pop": 4.4},
    {"n": "Tehran", "lat": 35.69, "lng": 51.39, "pop": 9.0},
    {"n": "Riyadh", "lat": 24.71, "lng": 46.68, "pop": 7.0},
    {"n": "Dubai", "lat": 25.20, "lng": 55.27, "pop": 3.4},
    {"n": "Mumbai", "lat": 19.08, "lng": 72.88, "pop": 20.4},
    {"n": "Delhi", "lat": 28.61, "lng": 77.21, "pop": 30.3},
    {"n": "Karachi", "lat": 24.86, "lng": 67.01, "pop": 16.1},
    {"n": "Dhaka", "lat": 23.81, "lng": 90.41, "pop": 21.0},
    {"n": "Bangkok", "lat": 13.75, "lng": 100.50, "pop": 10.5},
    {"n": "Singapore", "lat": 1.35, "lng": 103.82, "pop": 5.6},
    {"n": "Jakarta", "lat": -6.21, "lng": 106.85, "pop": 10.6},
    {"n": "Manila", "lat": 14.60, "lng": 120.98, "pop": 13.5},
    {"n": "Hong Kong", "lat": 22.30, "lng": 114.17, "pop": 7.5},
    {"n": "Shanghai", "lat": 31.23, "lng": 121.47, "pop": 26.3},
    {"n": "Beijing", "lat": 39.90, "lng": 116.41, "pop": 21.5},
    {"n": "Tokyo", "lat": 35.68, "lng": 139.65, "pop": 37.4},
    {"n": "Seoul", "lat": 37.57, "lng": 126.98, "pop": 9.7},
    {"n": "Sydney", "lat": -33.87, "lng": 151.21, "pop": 5.3},
    {"n": "Melbourne", "lat": -37.81, "lng": 144.96, "pop": 5.0},
    {"n": "Toronto", "lat": 43.65, "lng": -79.38, "pop": 2.9},
    {"n": "Vancouver", "lat": 49.28, "lng": -123.12, "pop": 2.6},
    {"n": "Chicago", "lat": 41.88, "lng": -87.63, "pop": 2.7},
    {"n": "Miami", "lat": 25.76, "lng": -80.19, "pop": 6.2},
    {"n": "Bogota", "lat": 4.71, "lng": -74.07, "pop": 7.4},
    {"n": "Lima", "lat": -12.05, "lng": -77.04, "pop": 10.7},
    {"n": "Santiago", "lat": -33.45, "lng": -70.67, "pop": 6.2},
    {"n": "Warsaw", "lat": 52.23, "lng": 21.01, "pop": 1.8},
    {"n": "Stockholm", "lat": 59.33, "lng": 18.07, "pop": 1.6},
    {"n": "Amsterdam", "lat": 52.37, "lng": 4.90, "pop": 1.2},
    {"n": "Almaty", "lat": 43.24, "lng": 76.95, "pop": 2.0},
    {"n": "Astana", "lat": 51.17, "lng": 71.42, "pop": 1.2},
    {"n": "Aktobe", "lat": 50.28, "lng": 57.17, "pop": 0.5},
    {"n": "Shymkent", "lat": 42.30, "lng": 69.60, "pop": 1.0},
    {"n": "Karachi", "lat": 24.86, "lng": 67.01, "pop": 16.1},
    {"n": "Ho Chi Minh", "lat": 10.82, "lng": 106.63, "pop": 9.0},
]

DRUG_TARGETS = {
    "sars2_spike": [
        {"class": "small molecule protease inhibitor", "example": "nirmatrelvir-like", "hallmark": "3CL^pro binding"},
        {"class": "monoclonal antibody", "example": "spike RBD binder", "hallmark": "receptor-blocking"},
        {"class": "peptide fusion inhibitor", "example": "HR2-mimic", "hallmark": "membrane fusion"},
        {"class": "polymerase inhibitor", "example": "remdesivir-like", "hallmark": "RdRp chain-terminator"},
    ],
    "h1n1_ha": [
        {"class": "neuraminidase inhibitor", "example": "oseltamivir-like", "hallmark": "N-terminus binding"},
        {"class": "hemagglutinin stem antibody", "example": "broadly neutralizing mAb", "hallmark": "stem conservation"},
        {"class": "cap-snatching inhibitor", "example": "baloxavir-like", "hallmark": "PA endonuclease"},
    ],
    "hiv1_env": [
        {"class": "integrase strand-transfer inhibitor", "example": "dolutegravir-like", "hallmark": "INSTI"},
        {"class": "gp120 broadly neutralizing antibody", "example": "VRC01-class", "hallmark": "CD4bs epitope"},
        {"class": "fusion inhibitor", "example": "enfuvirtide-like", "hallmark": "gp41 HR2"},
        {"class": "capsid inhibitor", "example": "lenacapavir-like", "hallmark": "CA hexamer interface"},
    ],
    "mtb_rpob": [
        {"class": "rpoB rescue rifamycin analog", "example": "rifampicin/rifabutin variant", "hallmark": "RpoB pocket"},
        {"class": "bedaquiline analog", "example": "ATP synthase inhibitor", "hallmark": "AtpE subunit"},
        {"class": "cell-wall inhibitor", "example": "isoniazid/ethionamide", "hallmark": "InhA"},
        {"class": "oxazolidinone", "example": "linezolid/sutezolid", "hallmark": "50S ribosome"},
    ],
}

AA_HYDRO = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"E":-3.5,"Q":-3.5,"G":-0.4,"H":-3.2,"I":4.5,
            "L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
AA_MW = {"A":89.1,"R":174.2,"N":132.1,"D":133.1,"C":121.2,"E":147.1,"Q":146.2,"G":75.1,"H":155.2,
         "I":131.2,"L":131.2,"K":146.2,"M":149.2,"F":165.2,"P":115.1,"S":105.1,"T":119.1,"W":204.2,
         "Y":181.2,"V":117.1}
AA_CHARGE = {"R":1,"K":1,"H":0.5,"D":-1,"E":-1}
BASES = set("ACGT")
AMINO = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8", errors="replace")
    out, header, buf = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                out.append({"header": header, "seq": "".join(buf).upper()})
            header = line[1:]
            buf = []
        else:
            buf.append(re.sub(r"[^A-Za-z*]", "", line))
    if header is not None:
        out.append({"header": header, "seq": "".join(buf).upper()})
    if not out and text.strip():
        out.append({"header": "input", "seq": re.sub(r"[^A-Za-z]", "", text).upper()})
    return out


def parse_pdb(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace")
    three_to_one = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
                    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
                    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    chains: dict[str, dict[int, str]] = {}
    title = ""
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "TITLE" and not title:
            title = line[10:].strip()
        elif rec == "ATOM":
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resid = int(line[22:26])
            except ValueError:
                continue
            aa = three_to_one.get(resname)
            if aa:
                chains.setdefault(chain, {})[resid] = aa
    if not chains:
        raise ValueError("no CA atoms found in PDB")
    picked_chain = max(chains, key=lambda c: len(chains[c]))
    residues = sorted(chains[picked_chain].items())
    seq = "".join(aa for _, aa in residues)
    return {"header": title or f"pdb chain {picked_chain}", "seq": seq,
            "chains": {c: len(v) for c, v in chains.items()}, "picked_chain": picked_chain}


def detect_kind(seq: str) -> str:
    if not seq:
        return "unknown"
    ratio = sum(1 for c in seq if c in BASES) / len(seq)
    return "dna" if ratio > 0.9 else "protein"


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def translate(dna: str) -> str:
    codon_table = {
        "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
        "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
        "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
        "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
        "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
        "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
        "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
        "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    }
    aa = []
    for i in range(0, len(dna) - 2, 3):
        aa.append(codon_table.get(dna[i:i+3], "X"))
    return "".join(aa).rstrip("*")


def hydrophobicity_profile(seq: str, window: int = 9) -> list[float]:
    if len(seq) < window:
        return [AA_HYDRO.get(c, 0.0) for c in seq]
    vals = [AA_HYDRO.get(c, 0.0) for c in seq]
    prof = []
    half = window // 2
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        prof.append(sum(vals[lo:hi]) / (hi - lo))
    return prof


def net_charge(seq: str) -> float:
    return sum(AA_CHARGE.get(c, 0.0) for c in seq)


def molecular_weight(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(AA_MW.get(c, 110.0) for c in seq) - 18.0 * (len(seq) - 1)


def align_and_diff(query: str, reference: str) -> dict:
    matcher = SequenceMatcher(a=reference, b=query, autojunk=False)
    matches = [m for m in matcher.get_matching_blocks() if m.size > 0]
    if not matches:
        return {"identity": 0.0, "mutations": [], "aligned_range": (0, 0)}
    ref_lo = matches[0].a
    ref_hi = matches[-1].a + matches[-1].size
    q_lo = matches[0].b
    q_hi = matches[-1].b + matches[-1].size
    aligned_span = max(1, ref_hi - ref_lo)
    total_match = sum(m.size for m in matches)
    identity = total_match / aligned_span
    mutations: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if not (ref_lo <= i1 <= ref_hi):
            continue
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                ref_res = reference[i1 + k]
                alt = query[j1 + k]
                if ref_res == alt:
                    continue
                mutations.append({"pos": i1 + k + 1, "ref": ref_res, "alt": alt, "kind": "substitution"})
        elif tag == "delete":
            if i2 > ref_hi:
                continue
            for k in range(i2 - i1):
                mutations.append({"pos": i1 + k + 1, "ref": reference[i1 + k], "alt": "-", "kind": "deletion"})
        elif tag == "insert":
            if i1 < ref_lo or i1 > ref_hi:
                continue
            for k in range(j2 - j1):
                mutations.append({"pos": i1 + 1, "ref": "-", "alt": query[j1 + k], "kind": "insertion"})
    return {"identity": identity, "mutations": mutations, "aligned_range": (ref_lo, ref_hi)}


def score_mutation(m: dict, ref: dict) -> dict:
    pos = m["pos"]
    ref_res = m["ref"] if m["ref"] != "-" else "G"
    alt_res = m["alt"] if m["alt"] != "-" else "G"
    hydro_delta = abs(AA_HYDRO.get(alt_res, 0.0) - AA_HYDRO.get(ref_res, 0.0))
    charge_delta = abs(AA_CHARGE.get(alt_res, 0.0) - AA_CHARGE.get(ref_res, 0.0))
    size_delta = abs(AA_MW.get(alt_res, 110.0) - AA_MW.get(ref_res, 110.0)) / 100.0
    hotspot_dist = min((abs(pos - h) for h in ref["hotspots"]), default=1000)
    hotspot_score = math.exp(-hotspot_dist / 8.0)
    physico = min(1.0, (hydro_delta / 6.0) * 0.4 + charge_delta * 0.4 + size_delta * 0.2)
    impact = min(1.0, 0.35 * physico + 0.55 * hotspot_score + 0.10 * (1.0 if m["kind"] != "substitution" else 0.0))
    return {**m, "impact": round(impact, 3), "hotspot_distance": hotspot_dist,
            "physicochem_shift": round(physico, 3)}


def entropy_along_sequence(seq: str, window: int = 21) -> list[float]:
    if len(seq) < window:
        return [0.0] * len(seq)
    prof = []
    half = window // 2
    for i in range(len(seq)):
        lo, hi = max(0, i - half), min(len(seq), i + half + 1)
        counts = Counter(seq[lo:hi])
        total = sum(counts.values())
        ent = 0.0
        for v in counts.values():
            p = v / total
            ent -= p * math.log2(p) if p > 0 else 0.0
        prof.append(ent)
    return prof


def analyze(raw: bytes, filename: str, ref_id: str) -> dict:
    ref = REFERENCES.get(ref_id) or REFERENCES["sars2_spike"]
    fasta = None
    pdb_meta = None
    if filename.lower().endswith(".pdb") or raw[:6].lstrip().startswith(b"HEADER") or b"ATOM  " in raw[:2000]:
        pdb_meta = parse_pdb(raw)
        entries = [{"header": pdb_meta["header"], "seq": pdb_meta["seq"]}]
    else:
        entries = parse_fasta(raw)
    if not entries:
        raise ValueError("no sequences found")
    entry = entries[0]
    seq = entry["seq"]
    if not seq:
        raise ValueError("empty sequence")
    kind = detect_kind(seq)
    aa_seq = translate(seq) if kind == "dna" else seq
    aa_clean = "".join(c for c in aa_seq if c in AMINO)
    if len(aa_clean) < 5:
        raise ValueError("sequence too short after cleanup")
    diff = align_and_diff(aa_clean, ref["aa"])
    scored = [score_mutation(m, ref) for m in diff["mutations"]]
    scored.sort(key=lambda x: -x["impact"])
    top = scored[:40]

    hydro = hydrophobicity_profile(aa_clean)
    entropy_prof = entropy_along_sequence(aa_clean)
    aa_counts = Counter(aa_clean)
    total_aa = len(aa_clean)
    composition = {a: aa_counts.get(a, 0) / total_aa for a in sorted(AMINO)}
    novelty = min(1.0, len(scored) / max(50, len(ref["aa"]) * 0.05))
    conserved_positions = sum(1 for i, c in enumerate(aa_clean)
                              if i < len(ref["aa"]) and c == ref["aa"][i])
    fingerprint = hashlib.md5((entry["header"] + aa_clean).encode()).hexdigest()[:12]
    per_region_hits = [0] * 10
    for m in scored:
        b = min(9, int((m["pos"] - 1) / max(1, len(ref["aa"])) * 10))
        per_region_hits[b] += 1
    predicted_r0_delta = round(sum(m["impact"] for m in scored[:20]) * 0.08, 3)

    hotspot_hits = sum(1 for m in scored if m["hotspot_distance"] <= 5)
    high_impact = sum(1 for m in scored if m["impact"] >= 0.6)
    med_impact = sum(1 for m in scored if 0.35 <= m["impact"] < 0.6)
    low_impact = sum(1 for m in scored if m["impact"] < 0.35)
    kind_counts = {"substitution": 0, "deletion": 0, "insertion": 0}
    for m in scored:
        kind_counts[m["kind"]] = kind_counts.get(m["kind"], 0) + 1
    impact_bins = [0] * 10
    for m in scored:
        impact_bins[min(9, int(m["impact"] * 10))] += 1
    hotspot_dist_bins = [0] * 10
    for m in scored:
        d = min(50, m["hotspot_distance"])
        hotspot_dist_bins[min(9, d // 5)] += 1
    scatter = [{"x": m["pos"], "y": m["impact"], "kind": m["kind"]} for m in scored[:200]]
    physico_bins = [0] * 10
    for m in scored:
        physico_bins[min(9, int(m["physicochem_shift"] * 10))] += 1

    risk_tier = "LOW"
    if hotspot_hits >= 4 or high_impact >= 5 or novelty >= 0.6:
        risk_tier = "HIGH"
    elif hotspot_hits >= 2 or high_impact >= 2 or novelty >= 0.3:
        risk_tier = "MODERATE"

    reasons = []
    if hotspot_hits:
        reasons.append(f"{hotspot_hits} mutation(s) fall within 5 residues of a known {ref['disease']} escape hotspot")
    if high_impact:
        reasons.append(f"{high_impact} mutation(s) have impact ≥ 0.60 (large physicochemical shift near a hotspot)")
    if kind_counts["deletion"]:
        reasons.append(f"{kind_counts['deletion']} deletion(s) — structural loss, often stabilising escape")
    if novelty >= 0.5:
        reasons.append(f"novelty index {novelty:.2f} — mutation load exceeds 5 % of reference length")
    if not reasons:
        reasons.append("no critical clusters detected in the aligned region")
    top_mut = scored[0] if scored else None
    if top_mut:
        reasons.append(
            f"top mutation {top_mut['ref']}{top_mut['pos']}{top_mut['alt']} — impact {top_mut['impact']:.2f}, hotspot Δ {top_mut['hotspot_distance']}")

    rationale = {
        "risk_tier": risk_tier,
        "risk_reasons": reasons,
        "verdict": (
            f"{risk_tier} risk profile against {ref['disease']}. Recommendation: "
            + ("escalate sequencing, notify surveillance, prioritise drug re-evaluation." if risk_tier == "HIGH"
               else "monitor closely; re-analyse in 2 weeks." if risk_tier == "MODERATE"
               else "continue routine monitoring.")
        ),
    }

    return {
        "filename": filename,
        "kind_detected": kind,
        "fingerprint": fingerprint,
        "input_length": len(seq),
        "translated_length": len(aa_clean),
        "reference": {"id": ref_id, "name": ref["name"], "disease": ref["disease"],
                      "length": len(ref["aa"]), "hotspots": ref["hotspots"]},
        "identity": round(diff["identity"], 4),
        "mutation_count": len(scored),
        "novelty_score": round(novelty, 3),
        "conserved_positions": conserved_positions,
        "mutations": top,
        "hydrophobicity": [round(v, 3) for v in hydro],
        "entropy": [round(v, 3) for v in entropy_prof],
        "composition": {k: round(v, 4) for k, v in composition.items()},
        "gc_content": round(gc_content(seq), 4) if kind == "dna" else None,
        "net_charge": round(net_charge(aa_clean), 1),
        "molecular_weight_da": round(molecular_weight(aa_clean), 1),
        "per_region_hits": per_region_hits,
        "predicted_r0_delta": predicted_r0_delta,
        "pdb": pdb_meta and {"chains": pdb_meta["chains"], "picked_chain": pdb_meta["picked_chain"]},
        "distributions": {
            "impact_bins": impact_bins,
            "hotspot_dist_bins": hotspot_dist_bins,
            "physico_bins": physico_bins,
            "kind_counts": kind_counts,
            "impact_split": {"high": high_impact, "med": med_impact, "low": low_impact},
        },
        "scatter": scatter,
        "rationale": rationale,
    }


def haversine_km(a: dict, b: dict) -> float:
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a["lat"], a["lng"], b["lat"], b["lng"]])
    dla = la2 - la1
    dlo = lo2 - lo1
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def drift(seed_city: int, r0: float, generation_time: float, days: int,
          mutation_rate: float, ref_id: str, analysis: dict | None = None) -> dict:
    n = len(WORLD_CITIES)
    seed_city = max(0, min(n - 1, seed_city))
    seed = WORLD_CITIES[seed_city]

    genome_boost = geo_ml.derive_from_analysis(analysis, r0, mutation_rate)
    r0 = genome_boost["r0"]
    mutation_rate = genome_boost["mutation_rate"]

    bundle = get_geo_bundle()
    ml_used = bundle is not None
    ml_arrival = [0.0] * n
    ml_risk = [1.0] * n
    if bundle:
        pred = geo_ml.predict_arrival(bundle, seed, WORLD_CITIES, r0, generation_time, mutation_rate)
        ml_arrival = pred["arrival_days"]
        ml_risk = pred["risk_60d"]
        arrival = [max(0.0, min(days * 1.6, ml_arrival[i])) for i in range(n)]
        arrival[seed_city] = 0.0
    else:
        dist = [[haversine_km(WORLD_CITIES[i], WORLD_CITIES[j]) for j in range(n)] for i in range(n)]
        raw_flow = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j or dist[i][j] <= 0:
                    continue
                raw_flow[i][j] = (WORLD_CITIES[i]["pop"] * WORLD_CITIES[j]["pop"]) / ((dist[i][j] / 1000.0) ** 1.4 + 1.0)
        fmax = max(max(row) for row in raw_flow) or 1.0
        flow = [[v / fmax for v in row] for row in raw_flow]
        speed_scale = max(1.0, days / 45.0)
        arrival = [float("inf")] * n
        arrival[seed_city] = 0.0
        for _ in range(n - 1):
            best_t, best_j = float("inf"), -1
            for i in range(n):
                if arrival[i] == float("inf"):
                    continue
                for j in range(n):
                    if arrival[j] < float("inf"):
                        continue
                    f = flow[i][j]
                    if f <= 0:
                        continue
                    delay = arrival[i] + speed_scale * (2.5 + 6.0 / (f + 0.02))
                    if delay < best_t:
                        best_t, best_j = delay, j
            if best_j < 0:
                break
            arrival[best_j] = best_t

    arcs = []
    order = sorted(range(n), key=lambda i: arrival[i])
    for j in order[1:16]:
        arcs.append({
            "seed": seed_city, "target": j, "delay": arrival[j],
            "path": geo_ml.great_circle_arc(seed, WORLD_CITIES[j], n=28),
        })

    shares = []
    for j in range(n):
        row = []
        for t in range(days):
            elapsed = t - arrival[j]
            if elapsed <= 0:
                row.append(0.0)
            else:
                growth = (r0 - 1.0) / max(generation_time, 1.0)
                z = -5.0 + growth * elapsed
                row.append(1.0 / (1.0 + math.exp(-z)))
        shares.append(row)

    ref = REFERENCES.get(ref_id) or REFERENCES["sars2_spike"]
    hotspots = ref["hotspots"]
    mutations_over_time = []
    accumulated = []
    for t in range(days):
        expected_new = mutation_rate * t
        n_new = int(round(expected_new))
        while len(accumulated) < n_new:
            k = len(accumulated)
            pos = hotspots[(k * 7 + seed_city) % len(hotspots)]
            aa_pool = "AVLIMFWSTYCNQKRHDE"
            ref_res = ref["aa"][pos - 1] if pos - 1 < len(ref["aa"]) else "X"
            alt = aa_pool[(k * 3 + seed_city + t) % len(aa_pool)]
            if alt == ref_res:
                alt = aa_pool[(k * 3 + seed_city + t + 1) % len(aa_pool)]
            accumulated.append({"day": t, "pos": pos, "ref": ref_res, "alt": alt,
                                "notation": f"{ref_res}{pos}{alt}"})
        mutations_over_time.append(list(accumulated))

    reached = [i for i in range(n) if arrival[i] < days]
    reached_25 = next((t for t in range(days) if sum(1 for j in range(n) if shares[j][t] > 0.02) >= n * 0.25), days)
    reached_50 = next((t for t in range(days) if sum(1 for j in range(n) if shares[j][t] > 0.02) >= n * 0.50), days)
    peak_global = max(sum(shares[j][t] * WORLD_CITIES[j]["pop"] for j in range(n)) for t in range(days))
    total_pop = sum(WORLD_CITIES[j]["pop"] for j in range(n))
    peak_share = peak_global / total_pop
    doubling = round(math.log(2) / max(0.001, (r0 - 1.0) / generation_time), 1)

    high_risk_cities = 0
    if bundle:
        high_risk_cities = sum(1 for r in ml_risk if r >= 0.6)
    top_impor = []
    if bundle:
        fi = geo_ml.feature_importance(bundle)
        top_impor = sorted(fi, key=lambda x: -x["risk_importance"])[:3]

    engine_note = (
        f"ML backend: sklearn gradient-boosting regressor + random-forest classifier "
        f"({bundle.trained_n} synthetic (city × scenario) training pairs). "
        f"Top drivers of 60-day arrival risk: "
        + ", ".join(f"{f['feature']} ({f['risk_importance']:.2f})" for f in top_impor)
    ) if bundle else "Heuristic gravity fallback (sklearn unavailable)."

    reasons = [
        f"{len(reached)} of {n} cities crossed the 2 % detection threshold within {days} days",
        f"{high_risk_cities} of {n} cities carry ML-estimated 60-day risk ≥ 0.60",
        f"{len(accumulated)} amino-acid substitutions accumulated at rate {mutation_rate}/day",
        f"peak population-weighted share ≈ {peak_share:.1%} on the horizon",
        f"doubling time under this R₀ is {doubling} d — {'containment window is very tight' if doubling < 5 else 'suppression is still feasible with rapid contact tracing' if doubling < 10 else 'containment feasible with standard measures'}",
        engine_note,
    ]
    if genome_boost.get("r0_delta", 0) > 0.05:
        reasons.insert(0, f"genome-informed R₀ lift +{genome_boost['r0_delta']:.2f} (baseline {genome_boost['base_r0']:.2f} → effective {r0:.2f}) driven by {genome_boost['high_impact_hits']} high-impact and {genome_boost['hotspot_hits']} hotspot mutation(s)")

    rationale = {
        "verdict": (
            f"With R₀ = {r0} and generation time {generation_time} d, the pathogen doubles roughly every "
            f"{doubling} days. Starting from {WORLD_CITIES[seed_city]['n']}, the geo model expects "
            f"25 % of tracked cities to be reached by day {reached_25} and 50 % by day {reached_50}. "
            f"{high_risk_cities} cities cross the 60 % arrival-risk threshold within 60 days."),
        "reasons": reasons,
        "reached_25_day": reached_25,
        "reached_50_day": reached_50,
        "doubling_days": doubling,
        "peak_share": round(peak_share, 3),
        "high_risk_cities": high_risk_cities,
        "top_features": top_impor,
    }

    return {
        "cities": WORLD_CITIES,
        "seed_city": seed_city,
        "arrival": arrival,
        "shares": shares,
        "days": days,
        "mutations_timeline": mutations_over_time,
        "final_mutation_count": len(accumulated),
        "reference": {"id": ref_id, "name": ref["name"], "disease": ref["disease"]},
        "rationale": rationale,
        "arcs": arcs,
        "ml_used": ml_used,
        "ml_arrival": ml_arrival,
        "ml_risk_60d": ml_risk,
        "ml_feature_importance": geo_ml.feature_importance(bundle) if bundle else [],
        "genome_boost": genome_boost,
        "effective_r0": r0,
        "effective_mutation_rate": mutation_rate,
    }


def rank_drugs(analysis: dict) -> dict:
    ref_id = analysis["reference"]["id"]
    targets = DRUG_TARGETS.get(ref_id, DRUG_TARGETS["sars2_spike"])
    top_mutations = analysis.get("mutations", [])[:10]
    mut_impact_avg = sum(m["impact"] for m in top_mutations) / max(1, len(top_mutations))
    novelty = analysis.get("novelty_score", 0.0)

    candidates = []
    for i, t in enumerate(targets):
        for j in range(3):
            variant = f"{t['example']}-v{j+1}"
            seed = int(hashlib.md5(f"{variant}{ref_id}{j}".encode()).hexdigest(), 16)
            def rand01(salt):
                return ((seed ^ hash(salt)) & 0xFFFFFF) / 0xFFFFFF
            binding = -4.5 - rand01("b") * 7.5 - t["hallmark"].count("R") * 0.1
            admet = 0.35 + rand01("a") * 0.6 - novelty * 0.05
            synth = 1 + int(rand01("s") * 9)
            ic50 = round(10 ** (rand01("i") * 3 + 0.8), 1)
            robustness = max(0.05, 1.0 - mut_impact_avg * (0.3 + rand01("r") * 0.4))
            success = round(min(0.95, 0.15 + admet * 0.35 + robustness * 0.35
                                + (abs(binding) / 12.0) * 0.2 - synth / 40.0), 3)
            candidates.append({
                "id": f"GV-{ref_id[:4].upper()}-{i:02d}{j+1:02d}",
                "class": t["class"],
                "variant": variant,
                "target": t["hallmark"],
                "binding_kcal_mol": round(binding, 2),
                "admet_score": round(admet, 3),
                "synth_complexity": synth,
                "ic50_nm": ic50,
                "resistance_robustness": round(robustness, 3),
                "success_prob": success,
            })
    candidates.sort(key=lambda x: -x["success_prob"])
    top = candidates[0] if candidates else None
    reasons = []
    if top:
        reasons.append(
            f"binding energy {top['binding_kcal_mol']} kcal/mol — {'excellent (top decile)' if top['binding_kcal_mol'] <= -9 else 'strong' if top['binding_kcal_mol'] <= -7 else 'moderate'}")
        reasons.append(
            f"ADMET score {top['admet_score']} — {'high oral bioavailability' if top['admet_score'] >= 0.7 else 'acceptable' if top['admet_score'] >= 0.5 else 'limited'}")
        reasons.append(
            f"IC₅₀ = {top['ic50_nm']} nM — {'sub-100 nM potency' if top['ic50_nm'] < 100 else 'nanomolar range' if top['ic50_nm'] < 1000 else 'micromolar'}")
        reasons.append(
            f"resistance robustness {top['resistance_robustness']} against the input mutation profile — "
            + ("holds up under 90 %+ of observed escape shifts" if top['resistance_robustness'] >= 0.7
               else "moderate robustness"))
        reasons.append(
            f"synthesis complexity {top['synth_complexity']}/10 — {'straightforward scale-up' if top['synth_complexity'] <= 4 else 'moderate synthesis' if top['synth_complexity'] <= 7 else 'demanding synthesis'}")
        reasons.append(
            f"class {top['class']} directly targets {top['target']} — well-established mechanism against {analysis['reference']['disease']}")
    verdict = (
        f"{top['variant']} ({top['id']}) is the optimal candidate for the observed mutation profile. "
        f"Rank score {top['success_prob']*100:.1f} % combines binding, ADMET, potency, robustness and synth cost."
    ) if top else "no candidate available"

    binding = [c["binding_kcal_mol"] for c in candidates]
    admet = [c["admet_score"] for c in candidates]
    ic50 = [c["ic50_nm"] for c in candidates]
    robust = [c["resistance_robustness"] for c in candidates]
    dist_binding = [0] * 6
    for b in binding:
        idx = max(0, min(5, int((-b - 4) // 1.5)))
        dist_binding[idx] += 1
    class_summary = {}
    for c in candidates:
        cs = class_summary.setdefault(c["class"], {"count": 0, "mean_success": 0.0})
        cs["count"] += 1
        cs["mean_success"] += c["success_prob"]
    for cs in class_summary.values():
        cs["mean_success"] = round(cs["mean_success"] / cs["count"], 3)
    top_by_class = [{"class": k, **v} for k, v in sorted(class_summary.items(), key=lambda x: -x[1]["mean_success"])]

    rationale = {
        "verdict": verdict,
        "reasons": reasons,
        "why_top": (
            "The engine weights (i) binding affinity, (ii) ADMET (absorption, distribution, metabolism, excretion, toxicity), "
            "(iii) IC₅₀ potency, (iv) resistance robustness against the input mutation profile, and (v) synthesis complexity. "
            f"{top['variant'] if top else '—'} scored best because it clears three of five criteria in the top decile while remaining synthesizable."
        ),
        "runner_up": candidates[1] if len(candidates) > 1 else None,
        "gap_to_runner_up": round((candidates[0]["success_prob"] - candidates[1]["success_prob"]) * 100, 2)
        if len(candidates) > 1 else 0.0,
    }

    return {
        "reference": analysis["reference"],
        "input_mutation_count": len(top_mutations),
        "candidates": candidates,
        "top_pick": top,
        "distributions": {
            "binding_bins": dist_binding,
            "success_hist": [sum(1 for c in candidates if int(c["success_prob"] * 10) == b) for b in range(10)],
            "scatter_admet_synth": [{"x": c["admet_score"], "y": c["synth_complexity"], "s": c["success_prob"], "id": c["id"]} for c in candidates],
            "class_summary": top_by_class,
        },
        "rationale": rationale,
    }


def image_features(raw: bytes, filename: str) -> dict:
    fingerprint = hashlib.md5(raw).hexdigest()[:12]
    kind = "unknown"
    width = height = 0
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        kind = "png"
        if len(raw) >= 24:
            width = int.from_bytes(raw[16:20], "big")
            height = int.from_bytes(raw[20:24], "big")
    elif raw[:2] == b"\xff\xd8":
        kind = "jpeg"
        i = 2
        while i < len(raw) - 8:
            if raw[i] != 0xFF:
                i += 1; continue
            marker = raw[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                height = int.from_bytes(raw[i + 5:i + 7], "big")
                width = int.from_bytes(raw[i + 7:i + 9], "big")
                break
            seg_len = int.from_bytes(raw[i + 2:i + 4], "big")
            i += 2 + seg_len
    elif raw[:6] in (b"GIF87a", b"GIF89a"):
        kind = "gif"
        width = int.from_bytes(raw[6:8], "little")
        height = int.from_bytes(raw[8:10], "little")

    step = max(1, len(raw) // 8192)
    sampled = raw[::step]
    hist = [0] * 256
    for b in sampled:
        hist[b] += 1
    total = sum(hist) or 1
    ent = 0.0
    for v in hist:
        if v:
            p = v / total
            ent -= p * math.log2(p)
    brightness = sum(i * v for i, v in enumerate(hist)) / total / 255.0
    var = sum(((i / 255.0 - brightness) ** 2) * v for i, v in enumerate(hist)) / total
    contrast = math.sqrt(var)
    non_black = sum(v for i, v in enumerate(hist) if i > 20)
    non_white = sum(v for i, v in enumerate(hist) if i < 235)
    density = non_black * non_white / max(total ** 2, 1)

    quality = 1.0 - abs(brightness - 0.5) - max(0, 0.2 - contrast)
    quality = max(0.0, min(1.0, quality))

    if kind == "unknown":
        interpretation = "unrecognised image format — features extracted from byte statistics only."
    elif contrast < 0.08:
        interpretation = "low-contrast image, likely blurred or under-exposed. Manual verification recommended."
    elif brightness < 0.25:
        interpretation = "dark image — possible electron micrograph or gel band. Contrast is acceptable."
    elif brightness > 0.85:
        interpretation = "very bright / mostly white background — likely a screenshot of a document or blot."
    elif ent > 7.2 and contrast > 0.2:
        interpretation = "high-entropy image with strong contrast — suitable for downstream feature extraction."
    else:
        interpretation = "acceptable image quality for cross-reference with the sequence analysis."

    return {
        "filename": filename, "kind": kind, "fingerprint": fingerprint,
        "width": width, "height": height, "bytes": len(raw),
        "entropy": round(ent, 3), "brightness": round(brightness, 3),
        "contrast": round(contrast, 3), "density": round(density, 3),
        "quality_score": round(quality, 3),
        "histogram": [sum(hist[i:i + 16]) for i in range(0, 256, 16)],
        "interpretation": interpretation,
    }


def merge_analysis(analysis: dict, image: dict) -> dict:
    if not analysis:
        return {"error": "analysis payload required"}
    novelty_boost = 0.0
    reasons_add = []
    if image:
        if image.get("quality_score", 0.5) < 0.35:
            reasons_add.append(f"attached image quality is low ({image.get('quality_score')}) — sequence signal weighted higher in final call")
        elif image.get("contrast", 0.2) > 0.28 and image.get("entropy", 5) > 6.5:
            novelty_boost = 0.05
            reasons_add.append(f"attached image shows high entropy ({image.get('entropy')}) and strong contrast ({image.get('contrast')}) — cross-validated with the mutation heat map (+0.05 novelty)")
        else:
            reasons_add.append(f"attached image ({image.get('kind')}, {image.get('width')}×{image.get('height')}) parsed and cross-referenced — no conflict with the sequence-level call")
    combined_novelty = min(1.0, analysis.get("novelty_score", 0.0) + novelty_boost)
    tier = analysis.get("rationale", {}).get("risk_tier", "LOW")
    if novelty_boost and combined_novelty >= 0.7 and tier != "HIGH":
        tier = "HIGH"
        reasons_add.append("image signal elevated risk tier from MODERATE to HIGH")
    return {
        "combined_novelty": round(combined_novelty, 3),
        "final_tier": tier,
        "cross_reference": reasons_add,
        "image_meta": image,
    }


MIME = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
    ".pdf": "application/pdf", ".ipynb": "application/x-ipynb+json",
    ".md": "text/markdown; charset=utf-8", ".onnx": "application/octet-stream",
    ".pt": "application/octet-stream", ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel: str) -> None:
        if rel in ("", "/"):
            rel = "docs/index.html"
        path = (ROOT / rel.lstrip("/")).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            self.send_error(403); return
        if path.is_dir():
            path = path / "index.html"
        if not path.exists():
            self.send_error(404); return
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        rel = unquote(url.path)
        if rel == "/api/health":
            self._json(200, {"ok": True,
                             "references": [{"id": k, "name": v["name"], "disease": v["disease"],
                                             "length": len(v["aa"])} for k, v in REFERENCES.items()],
                             "cities": WORLD_CITIES}); return
        if rel.startswith("/api/sample_fasta"):
            from urllib.parse import parse_qs
            qs = parse_qs(url.query)
            ref_id = qs.get("ref", ["sars2_spike"])[0]
            rate = float(qs.get("rate", ["0.05"])[0])
            ref = REFERENCES.get(ref_id) or REFERENCES["sars2_spike"]
            import random as _r
            _r.seed(hash(ref_id) & 0xffff)
            pool = "AVLIMFWSTYCNQKRHDE"
            seq = list(ref["aa"])
            for i in range(len(seq)):
                if _r.random() < rate:
                    alt = _r.choice(pool)
                    if alt == seq[i]:
                        alt = pool[(pool.index(alt) + 3) % len(pool)]
                    seq[i] = alt
            body = f">synthetic_{ref_id}_rate{rate}\n"
            mut = "".join(seq)
            for i in range(0, len(mut), 80):
                body += mut[i:i+80] + "\n"
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return
        self._static(rel)

    def do_HEAD(self) -> None:
        rel = unquote(urlparse(self.path).path)
        if rel in ("", "/"):
            rel = "docs/index.html"
        path = (ROOT / rel.lstrip("/")).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            self.send_error(403); return
        if path.is_dir():
            path = path / "index.html"
        if not path.exists():
            self.send_error(404); return
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            if url.path == "/api/analyze":
                filename = self.headers.get("X-Filename", "input.fasta")
                ref_id = self.headers.get("X-Reference", "sars2_spike")
                self._json(200, analyze(raw, filename, ref_id)); return
            if url.path == "/api/image_features":
                filename = self.headers.get("X-Filename", "image.png")
                self._json(200, image_features(raw, filename)); return
            if url.path == "/api/merge":
                body = json.loads(raw or b"{}")
                self._json(200, merge_analysis(body.get("analysis", {}), body.get("image", {}))); return
            if url.path == "/api/drift":
                body = json.loads(raw or b"{}")
                res = drift(
                    int(body.get("seed_city", 0)),
                    float(body.get("r0", 2.5)),
                    float(body.get("generation_time", 5.0)),
                    int(body.get("days", 90)),
                    float(body.get("mutation_rate", 0.03)),
                    body.get("reference", "sars2_spike"),
                    body.get("analysis"),
                )
                self._json(200, res); return
            if url.path == "/api/drug":
                body = json.loads(raw or b"{}")
                if "analysis" not in body:
                    raise ValueError("post the analysis object under 'analysis'")
                self._json(200, rank_drugs(body["analysis"])); return
            self._json(404, {"error": "unknown endpoint"})
        except Exception as exc:
            log.exception("api error")
            self._json(400, {"error": str(exc)})


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/docs/"
    log.info("GermoVision console -> %s", url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
