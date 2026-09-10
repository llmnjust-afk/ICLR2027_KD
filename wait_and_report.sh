#!/bin/bash
echo "Started monitoring at $(date)"
while true; do
    done_count=0
    for cfg in cags entropy unguided; do
        if grep -q "Mean Best" /root/ICLR2027_KD/logs/eval_in200_${cfg}.log 2>/dev/null; then
            done_count=$((done_count + 1))
        fi
    done
    if [ $done_count -eq 3 ]; then
        echo "=== ALL P0-4 EVALS DONE at $(date) ==="
        for cfg in cags entropy unguided; do
            echo "--- ${cfg} ---"
            grep "Mean Best\|Seed.*Best Top-1" /root/ICLR2027_KD/logs/eval_in200_${cfg}.log 2>/dev/null
        done
        echo "=== D4M t40 ==="
        grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_in100_t40.log 2>/dev/null || echo "  Not done"
        echo "=== D4M Nette ==="
        grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_nette.log 2>/dev/null || echo "  Not done"
        echo "=== D4M Woof ==="
        grep "Mean Best" /root/ICLR2027_KD/logs/eval_d4m_woof.log 2>/dev/null || echo "  Not done"
        break
    fi
    echo "Waiting... P0-4 done: ${done_count}/3 at $(date)"
    sleep 120
done
