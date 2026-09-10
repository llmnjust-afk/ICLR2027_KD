#!/bin/bash
while true; do
    echo "===== $(date '+%H:%M:%S UTC') ====="
    echo "GPU0 IN-1K CAGS: $(tail -1 /root/ICLR2027_KD/logs/eval_in1k_cags.log)"
    echo "GPU1 IN-1K Ung:  $(tail -1 /root/ICLR2027_KD/logs/eval_in1k_unguided.log)"
    echo "GPU2 F101 s1:    $(tail -1 /root/ICLR2027_KD/logs/eval_sd_food101_unguided_seed1.log 2>/dev/null)"
    echo "GPU3 F101 s0:    $(tail -1 /root/ICLR2027_KD/logs/eval_sd_food101_unguided.log)"
    echo "Orch:            $(tail -1 /root/ICLR2027_KD/logs/orchestrate_gpus.log)"
    echo ""

    # Check if IN-1K CAGS finished
    if grep -q "Mean Best Top-1" /root/ICLR2027_KD/logs/eval_in1k_cags.log 2>/dev/null; then
        echo "!!! IN-1K CAGS COMPLETED !!!"
        grep "Mean Best" /root/ICLR2027_KD/logs/eval_in1k_cags.log
        break
    fi
    sleep 120
done
