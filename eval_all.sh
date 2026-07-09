#!/bin/bash
# One-click evaluation script: run all evaluation commands and save results to specified directory
# Usage: ./eval_all.sh <pkl_file> [--with_vina] [output_dir]
# Example: ./eval_all.sh samples_all.pkl
# Example: ./eval_all.sh samples_all.pkl --with_vina
# Example: ./eval_all.sh samples_all.pkl results/my_eval
# Example: ./eval_all.sh samples_all.pkl --with_vina results/my_eval

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <pkl_file> [--with_vina] [output_dir]"
    echo "Example: $0 samples_all.pkl"
    echo "Example: $0 samples_all.pkl --with_vina"
    echo "Example: $0 samples_all.pkl results/my_eval"
    echo "Example: $0 samples_all.pkl --with_vina results/my_eval"
    exit 1
fi

PKL_FILE="$1"
shift

# Parse arguments (Vina is skipped by default)
SKIP_VINA="--skip_vina"
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --with_vina)
            SKIP_VINA=""
            shift
            ;;
        --skip_vina)
            SKIP_VINA="--skip_vina"
            shift
            ;;
        *)
            OUTPUT_DIR="$1"
            shift
            ;;
    esac
done

# Check if pkl file exists
if [ ! -f "$PKL_FILE" ]; then
    echo "Error: File '$PKL_FILE' does not exist"
    exit 1
fi

# Get absolute path of pkl file
PKL_FILE=$(realpath "$PKL_FILE")
PKL_BASENAME=$(basename "$PKL_FILE" .pkl)
PKL_DIR=$(dirname "$PKL_FILE")

# Set output directory (default: same directory as pkl file)
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PKL_DIR}/eval_results_${PKL_BASENAME}_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo "======================================"
echo "Starting evaluation: $PKL_FILE"
echo "Results directory: $OUTPUT_DIR"
if [ -n "$SKIP_VINA" ]; then
    echo "Skip Vina docking: Yes"
else
    echo "Skip Vina docking: No"
fi
echo "======================================"

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Run evaluate.py
echo ""
echo "[1/5] Running evaluate.py (general property evaluation)..."
python evaluate.py --path "$PKL_FILE" --dataset crossdock $SKIP_VINA 2>&1 | tee "$OUTPUT_DIR/01_evaluate.txt"
echo "Done! Output saved to: $OUTPUT_DIR/01_evaluate.txt"

# 2. Run evaluate_collisions.py
echo ""
echo "[2/5] Running evaluate_collisions.py (collision frequency evaluation)..."
python scripts_for_eval/evaluate_collisions.py --sample_file "$PKL_FILE" 2>&1 | tee "$OUTPUT_DIR/02_evaluate_collisions.txt"
echo "Done! Output saved to: $OUTPUT_DIR/02_evaluate_collisions.txt"

# 3. Run filter_rdBrenk_simple.py
echo ""
echo "[3/5] Running filter_rdBrenk_simple.py (forbidden functional group evaluation)..."
python brenk_filter/filter_rdBrenk_simple.py "$PKL_FILE" 2>&1 | tee "$OUTPUT_DIR/03_filter_rdBrenk.txt"
echo "Done! Output saved to: $OUTPUT_DIR/03_filter_rdBrenk.txt"

# 4. Run check_ring_brenk.py
echo ""
echo "[4/5] Running check_ring_brenk.py (ring size evaluation)..."
python brenk_filter/check_ring_brenk.py -i "$PKL_FILE" 2>&1 | tee "$OUTPUT_DIR/04_check_ring_brenk.txt"
echo "Done! Output saved to: $OUTPUT_DIR/04_check_ring_brenk.txt"

# 5. Run calc_dihedral_kldiv.py
echo ""
echo "[5/5] Running calc_dihedral_kldiv.py (dihedral angle KL divergence evaluation)..."
python scripts_for_eval/calc_dihedral_kldiv.py --ligand_files "$PKL_FILE" --root_dir data/crossdocked_pocket10 2>&1 | tee "$OUTPUT_DIR/05_calc_dihedral_kldiv.txt"
echo "Done! Output saved to: $OUTPUT_DIR/05_calc_dihedral_kldiv.txt"

# 6. Generate summary file summary.txt
echo ""
echo "[Summary] Generating summary.txt..."

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

# Extract metrics
# From 01_evaluate.txt
QED=$(grep "mean qed:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean qed://' | tr -d ' ')
SA=$(grep "mean sa:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean sa://' | tr -d ' ')
LOGP=$(grep "mean logP:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean logP://' | tr -d ' ')
LIPINSKI=$(grep "mean Lipinski:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean Lipinski://' | tr -d ' ')
MW=$(grep "mean molecular weight:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean molecular weight://' | tr -d ' ')
DIVERSITY=$(grep "diversity:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/diversity://' | tr -d ' ')

# Vina (if not skipped)
if [ -z "$SKIP_VINA" ]; then
    VINA=$(grep "mean vina:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/mean vina://' | tr -d ' ')
    HIGH_AFFINITY=$(grep "high affinity rate:" "$OUTPUT_DIR/01_evaluate.txt" | tail -1 | sed 's/high affinity rate://' | tr -d ' ')
else
    VINA="N/A (skipped)"
    HIGH_AFFINITY="N/A (skipped)"
fi

# From 02_evaluate_collisions.txt: extract collision rate (divided by 100)
# Format: Samples with collisions: 237 (47.4%)
COLLISION_RATE_RAW=$(grep "Samples with collisions:" "$OUTPUT_DIR/02_evaluate_collisions.txt" | awk -F'[(%]' '{print $2}')
if [ -n "$COLLISION_RATE_RAW" ]; then
    COLLISION_RATE=$(awk "BEGIN {printf \"%.4f\", $COLLISION_RATE_RAW / 100}")
else
    COLLISION_RATE="N/A"
fi

# From 03_filter_rdBrenk.txt: extract forbidden functional group rate
BRENK_RATE=$(grep "Removal rate:" "$OUTPUT_DIR/03_filter_rdBrenk.txt" | tail -1 | awk '{print $NF}')

# From 04_check_ring_brenk.txt: extract ring info (divided by 100)
# Format: ring size 3: 0.031 (3.1%)
RING_3_RAW=$(grep "ring size 3:" "$OUTPUT_DIR/04_check_ring_brenk.txt" | awk -F'[(%]' '{print $2}')
if [ -n "$RING_3_RAW" ]; then
    RING_3=$(awk "BEGIN {printf \"%.4f\", $RING_3_RAW / 100}")
else
    RING_3="N/A"
fi
RING_4_RAW=$(grep "ring size 4:" "$OUTPUT_DIR/04_check_ring_brenk.txt" | awk -F'[(%]' '{print $2}')
if [ -n "$RING_4_RAW" ]; then
    RING_4=$(awk "BEGIN {printf \"%.4f\", $RING_4_RAW / 100}")
else
    RING_4="N/A"
fi
# Format: Percentage of ligands with ring size >= 9: 2.21% (10/452)
RING_9_PLUS_RAW=$(grep "Percentage of ligands with ring size >= 9:" "$OUTPUT_DIR/04_check_ring_brenk.txt" | awk '{for(i=1;i<=NF;i++) if($i ~ /%/) {gsub(/%/,"",$i); print $i; exit}}')
if [ -n "$RING_9_PLUS_RAW" ]; then
    RING_9_PLUS=$(awk "BEGIN {printf \"%.4f\", $RING_9_PLUS_RAW / 100}")
else
    RING_9_PLUS="N/A"
fi

# From 05_calc_dihedral_kldiv.txt: extract dihedral cccc range ratio (divided by 100)
# Format: Generated Set 1: 0.0763 (7.63%)
CCCC_RANGE_RAW=$(grep "cccc Range" "$OUTPUT_DIR/05_calc_dihedral_kldiv.txt" -A2 | grep "Generated Set 1:" | awk -F'[(%]' '{print $2}')
if [ -n "$CCCC_RANGE_RAW" ]; then
    CCCC_RANGE=$(awk "BEGIN {printf \"%.4f\", $CCCC_RANGE_RAW / 100}")
else
    CCCC_RANGE="N/A"
fi

# Write summary file
cat > "$SUMMARY_FILE" << EOF
======================================================
Evaluation Results Summary
======================================================
Evaluated file: $PKL_FILE
Evaluation time: $(date)
======================================================

[Basic Metrics]
  QED:                ${QED:-N/A}
  SA:                 ${SA:-N/A}
  logP:               ${LOGP:-N/A}
  Lipinski:           ${LIPINSKI:-N/A}
  Molecular Weight:   ${MW:-N/A}
  Diversity:          ${DIVERSITY:-N/A}

[Binding Affinity]
  Vina Score:         ${VINA:-N/A}
  High Affinity Rate: ${HIGH_AFFINITY:-N/A}

[Collision Detection]
  Collision Rate:     ${COLLISION_RATE:-N/A}

[Forbidden Functional Groups]
  Brenk Filter Rate:  ${BRENK_RATE:-N/A}

[Ring Structure]
  3-Membered Ring:    ${RING_3:-N/A}
  4-Membered Ring:    ${RING_4:-N/A}
  9+ Membered Ring:   ${RING_9_PLUS:-N/A}

[Dihedral Angle]
  cccc Anomalous Range Rate: ${CCCC_RANGE:-N/A}
  (24~156 deg or -156~-24 deg)

======================================================
EOF

echo "Summary file generated: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

echo ""
echo "======================================"
echo "All evaluations completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "======================================"
echo ""
echo "Result file list:"
ls -la "$OUTPUT_DIR"
