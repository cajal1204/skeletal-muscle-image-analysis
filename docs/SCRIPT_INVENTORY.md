# Script inventory

The 23 scripts below are organized by purpose in the README. Their contents are unchanged from the original files.

| Script | Purpose |
| --- | --- |
| [analyze_rfp_nuclei_and_emhc_by_fiber.py](../scripts/analyze_rfp_nuclei_and_emhc_by_fiber.py) | Combined greatest-overlap RFP assignment and mean-based eMHC scoring; CLI. |
| [assign_nuclei_to_fibers.py](../scripts/assign_nuclei_to_fibers.py) | DAPI assignment, central nucleation, eroded CSA, and fiber QC for dystrophin workflow. |
| [assign_nuclei_to_fibers_emhc.py](../scripts/assign_nuclei_to_fibers_emhc.py) | DAPI assignment and central nucleation with eMHC workflow QC settings. |
| [assign_rfp_nuclei_to_fibers_ipax3dko.py](../scripts/assign_rfp_nuclei_to_fibers_ipax3dko.py) | Centroid-only RFP assignment in batch folders. |
| [calculate_csas.py](../scripts/calculate_csas.py) | Single-image CSA without additional erosion. |
| [lookup_fiber_by_id.py](../scripts/lookup_fiber_by_id.py) | Interactive individual-fiber visualization; optional Windows Fiji launch. |
| [lookup_fiber_by_id_emhc.py](../scripts/lookup_fiber_by_id_emhc.py) | Interactive individual-fiber visualization; optional Windows Fiji launch. |
| [make_all_fibers_central_dystrophin_overlay.py](../scripts/make_all_fibers_central_dystrophin_overlay.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [make_all_fibers_central_emhc_overlay.py](../scripts/make_all_fibers_central_emhc_overlay.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [make_fiber_id_overlay.py](../scripts/make_fiber_id_overlay.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [make_fiber_id_overlay_emhc.py](../scripts/make_fiber_id_overlay_emhc.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [make_qc_overlay_on_emhc.py](../scripts/make_qc_overlay_on_emhc.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [make_qc_overlay_on_image.py](../scripts/make_qc_overlay_on_image.py) | QC or labeled overlay visualization for the indicated marker/workflow. |
| [mark_fiber_not_centrally_nucleated.py](../scripts/mark_fiber_not_centrally_nucleated.py) | Manual correction of central-nucleation assignments and summaries. |
| [mark_fiber_not_centrally_nucleated_emhc.py](../scripts/mark_fiber_not_centrally_nucleated_emhc.py) | Manual correction of central-nucleation assignments and summaries. |
| [measure_fibers.py](../scripts/measure_fibers.py) | Batch dystrophin-mask CSA, border and size flags, individual CSVs. |
| [score_dystrophin_by_fiber.py](../scripts/score_dystrophin_by_fiber.py) | Inner-band dystrophin intensity/coverage classification. |
| [score_emhc_by_fiber.py](../scripts/score_emhc_by_fiber.py) | Core mean-based eMHC classification with full-fiber fallback. |
| [summary.py](../scripts/summary.py) | Counts and CSA summaries from measure_fibers outputs. |
| [toggle_dystrophin_status.py](../scripts/toggle_dystrophin_status.py) | Interactive marker-classification override. |
| [toggle_emhc_status.py](../scripts/toggle_emhc_status.py) | Interactive marker-classification override. |
| [toggle_manual_fiber_exclusion.py](../scripts/toggle_manual_fiber_exclusion.py) | Interactive manual fiber exclusion and QC update. |
| [toggle_manual_fiber_exclusion_emhc.py](../scripts/toggle_manual_fiber_exclusion_emhc.py) | Interactive manual fiber exclusion and QC update. |
