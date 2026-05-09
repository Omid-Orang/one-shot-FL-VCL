## One-Shot Federated Class-Incremental Learning via Variational Feature Transfer

This repository presents a novel one-shot federated class-incremental learning framework for medical imaging using variational feature transfer.

The proposed approach enables:
- One-shot federated learning
- Continual adaptation to new classes
- Distribution-level synthetic replay
- Near-zero catastrophic forgetting
- Privacy-preserving medical imaging analysis

## Abstract

Federated learning (FL) enables privacy-preserving collaboration for medical image analysis across decentralized institutions but faces major challenges from non-IID data distributions, high communication overhead in multi-round protocols, and catastrophic forgetting when models must adapt to sequentially arriving tasks.

We propose a novel class-incremental continual learning model for a one-shot FL paradigm in which clients estimate class-conditional feature distributions via variational inference and transmit compact statistics to the server.

The server synthesizes feature embeddings and learns a global classifier without revisiting client data or requiring additional communication rounds.

Extensive experiments on multiple medical imaging benchmarks demonstrate strong performance with near-zero forgetting.

## Framework

![Framework](figures/framework.pdf)

