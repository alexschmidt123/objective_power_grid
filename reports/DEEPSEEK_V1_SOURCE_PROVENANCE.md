# V1-only source provenance

Foundation13681d7091189a4e68056e6a2864dc58f9d5700b. Original policy, trainer, encoder, simulator and evaluator files remain byte-identical. The new train function is mechanically derived ONLY from that original trainer; its SHA256 before adaptation is f69a2e0799f30a99d074dcc56199f1097c20f026850e2123571cf17bc5c03179. No IR,RC,SA orPIMoE training/architecture source is used. Reference weights are for frozen comparison/pilot diagnostics; main models start fresh seeds101/102. The new model is the original DeepSeekPolicy class with8434 parameters.
