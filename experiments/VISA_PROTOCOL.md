# Real industrial-image evaluation protocol

Fixed before examining scores: use the VisA release VisA_20220922.tar and the authors' one-class split CSV from amazon-science/spot-diff. Source: https://github.com/amazon-science/spot-diff and https://registry.opendata.aws/visa/. Citation: Zou et al., SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, ECCV 2022. CC BY 4.0. Dataset images are not committed.

Choose cashew, capsules and pcb1 beforehand to cover single-object, multiple-instance and circuit-board inspection. Preserve the official train/test allocation. Use only train normal images, split 80/20 fit/normal validation with seed 42, fit feature scaling on fit only, fix the threshold at normal-validation 95th percentile. Reject duplicate training images and any exact image overlap with test. This is a handcrafted descriptor baseline, not PatchCore.

Prespecified settings: (resize, memory rows) = (64, 256), (128, 1000), (256, 1000). Report all settings, never select a winner using test labels. Each uses the same normal-image partition. Report image AUROC/AP, false alerts, detection rate, raw per-image scores, median/p95 CPU latency and measured model file bytes. Compare an explicit global RGB mean-distance baseline fit on the same fit images with threshold from the same normal-validation images. No neural comparison or pixel metric is implied. Timing is single-host, one CPU thread and includes descriptor extraction plus nearest-neighbor construction/query; not camera throughput.

Bootstrap image AUROC over 300 resampled test-image sets (fixed seed), skipping one-class replicates. These descriptive intervals do not capture dependence across image acquisitions or model-training variability. No field/factory validation claim.

## Follow-up neural comparison

Added after observing the handcrafted baseline's weaknesses. This is exploratory on the same test categories, not a new blind benchmark. Fix frozen torchvision ResNet18 IMAGENET1K_V1, resize 128, layers 2 and 3 with local 3x3 average pooling and layer-2 pooling to layer-3 resolution; concatenate 384-channel features. Train-only standardization, seeded random 1,000-patch memory (no coreset), unchanged fit/validation partition and 95th-percentile normal threshold. No weight finetuning. This is an attributed patch-neighbor baseline inspired by the family of pretrained patch methods, not a reproduction or invention of PatchCore. Report the result even if worse. Full deployment depends on official pretrained weights and optional torch dependencies.
