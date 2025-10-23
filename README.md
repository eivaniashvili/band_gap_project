# Matbench MP Gap — Composition-based Band Gap Prediction

This repository contains a reproducible notebook pipeline to:
1) load the Matbench MP Gap dataset,
2) perform EDA,
3) apply a grouped + stratified split (by chemical system and band-gap deciles),
4) featurize compositions with Matminer (ElementFraction + Magpie),
5) save clean Train/Val/Test matrices for classical ML.
