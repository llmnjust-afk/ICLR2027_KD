#!/bin/bash
echo "=== $(date) ==="
echo "--- P0-4 Results ---"
for cfg in cags entropy unguided; do
    echo "  ${cfg}:"
    grep "Mean Best" /root/ICLR2027_KD/logs/eval_in200_${cfg}.log 2>/dev/null || echo "    Not done yet"
    grep "Seed.*Best Top-1" /root/ICLR2027_KD/logs/eval_in200_${cfg}.log 2>/dev/null
done
echo "--- D4M t40 ---"
grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_in100_t40.log 2>/dev/null || echo "  Not done yet"
echo "--- D4M Nette ---"
grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_nette.log 2>/dev/null || echo "  Not done yet"
echo "--- D4M Woof ---"
grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_woof.log 2>/dev/null || echo "  Not done yet"
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
