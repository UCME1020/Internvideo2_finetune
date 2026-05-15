#!/usr/bin/env bash
# Sequential full finetune for all 4 datasets.
# Order: didemo → msrvtt → lsmdc → activitynet
#
# Each finetune writes its own logs under:
#   InternVideo/InternVideo2/multi_modality/scripts/finetuning/stage2/6B/<dataset>_blim/run.sh_<TS>/
#
# Status log: run_all_status.log (this dir)
#
# A single dataset failure does NOT abort the queue.

set -uo pipefail

KIT_DIR="/data5/ucjung/PoLaRT/Internvideo2_finetune"
STATUS_LOG="${KIT_DIR}/run_all_status.log"
ORDER=(didemo msrvtt lsmdc activitynet)

echo "=== run_all_finetunes started $(date -Iseconds) ===" | tee -a "${STATUS_LOG}"
echo "Order: ${ORDER[*]}" | tee -a "${STATUS_LOG}"
echo "Each: 4 GPU × 5 epoch, ~5-6h" | tee -a "${STATUS_LOG}"

for DS in "${ORDER[@]}"; do
    START_TS="$(date -Iseconds)"
    START_EPOCH="$(date +%s)"
    LOG="${KIT_DIR}/finetune_${DS}.log"
    echo "" | tee -a "${STATUS_LOG}"
    echo "── [${START_TS}] start: ${DS} → ${LOG}" | tee -a "${STATUS_LOG}"

    bash "${KIT_DIR}/train_iv2_6b.sh" "${DS}" > "${LOG}" 2>&1
    RC=$?

    END_TS="$(date -Iseconds)"
    DUR=$(( $(date +%s) - START_EPOCH ))
    DUR_H=$(( DUR / 3600 ))
    DUR_M=$(( (DUR % 3600) / 60 ))
    if [[ ${RC} -eq 0 ]]; then
        echo "── [${END_TS}] DONE  : ${DS} (rc=0, ${DUR_H}h${DUR_M}m)" | tee -a "${STATUS_LOG}"
    else
        echo "── [${END_TS}] FAIL  : ${DS} (rc=${RC}, ${DUR_H}h${DUR_M}m) — see ${LOG}" | tee -a "${STATUS_LOG}"
    fi
done

echo "" | tee -a "${STATUS_LOG}"
echo "=== run_all_finetunes finished $(date -Iseconds) ===" | tee -a "${STATUS_LOG}"
