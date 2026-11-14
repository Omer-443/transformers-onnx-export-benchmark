# Exploring the Transformers ONNX Exporter: CPU Inference and Graph Fusion

A practical experiment exploring the ONNX export workflow introduced in Hugging Face Transformers 5.23.0.

The goal was to compare PyTorch inference with standard ONNX Runtime and CPU-fused ONNX Runtime, while checking numerical consistency and inspecting the exported computation graphs.

The main question was simple:

**Does graph optimization necessarily translate into faster inference?**

## Experiment Overview

The experiment uses [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and compares three execution paths:

1. PyTorch
2. Standard ONNX Runtime
3. CPU-fused ONNX Runtime

It also examines embedding consistency, retrieval ranking consistency, and structural changes introduced by graph fusion.

## Environment

| Component | Version / Configuration |
|---|---|
| Python | 3.13.18 |
| PyTorch | 2.11.0+cpu |
| Transformers | 5.23.0 |
| ONNX Runtime | 1.32.0 |
| ONNX | 1.23.2 |
| Device | CPU |
| Logical CPU cores | 4 |

The experiment was conducted in a Kaggle notebook.

## What Was Tested

### 1. PyTorch Baseline

Loaded the model using the Transformers API and generated embeddings using attention-mask-aware mean pooling.

The embedding dimension was 384.

The model's Sentence Transformers configuration was also inspected to understand its intended processing pipeline:

`Transformer → Mean Pooling → L2 Normalization`

### 2. ONNX Export

Exported the model using the Transformers 5.23.0 exporter workflow.

Two ONNX variants were created:

- Standard ONNX export
- CPU-fused ONNX export using `ort_fusions="cpu"`

Both exported models were successfully loaded and executed using ONNX Runtime.

### 3. Numerical Consistency

Compared normalized PyTorch and ONNX embeddings.

Results:

| Metric | Result |
|---|---:|
| Maximum absolute difference | 1.64 × 10⁻⁷ |
| Mean absolute difference | 3.48 × 10⁻⁸ |
| Mean cosine similarity | 1.000000 |
| Minimum cosine similarity | 0.99999988 |

These results indicate very close numerical agreement for the tested inputs.

### 4. Retrieval Ranking Consistency

Used four test queries against a small four-item corpus.

- All four queries produced identical PyTorch and ONNX rankings.
- Top-k overlap was 100%.
- Spearman rank correlation was 1.0.

This is a small-scale implementation consistency check, not an evaluation of retrieval quality on a representative dataset.

### 5. ONNX Graph Inspection

Inspected the exported computation graphs to identify structural changes introduced by CPU fusion.

| Measurement | Standard ONNX | CPU-Fused ONNX |
|---|---:|---:|
| Graph nodes | 287 | 242 |
| Model file size | 0.581 MB | 0.307 MB |
| Microsoft-domain operators | 0 | 18 |

The fused graph contained:

- 6 `BiasGelu` operators
- 12 `SkipLayerNormalization` operators

This represents approximately 15.7% fewer graph nodes and 47.2% smaller serialized model size.

### 6. Controlled CPU Benchmark

The controlled benchmark used:

- 10 warm-up iterations per backend
- 100 measured iterations per round
- 5 rounds
- 4 PyTorch intra-op threads
- 4 ONNX Runtime intra-op threads
- 1 ONNX Runtime inter-op thread

The benchmark measured the transformer forward pass only. Tokenization, pooling, normalization, model loading, and export time were excluded.

| Backend | Median latency | Mean latency |
|---|---:|---:|
| PyTorch | 172.62 ms | 153.19 ms |
| Standard ONNX | 44.08 ms | 49.62 ms |
| CPU-Fused ONNX | 46.71 ms | 49.20 ms |

**Important:** These measurements showed substantial variability. The five rounds were performed within one process, not in five independent sessions.

An earlier exploratory benchmark produced substantially different timings. The two benchmark results should not be treated as directly comparable or used to establish a reliable speedup.

The controlled measurements did not establish a latency advantage for CPU fusion over standard ONNX.

## Key Observations

- The Transformers exporter successfully exported both ONNX variants.
- PyTorch and ONNX produced numerically consistent embeddings for the tested inputs.
- The fused graph contained identifiable fused operators.
- CPU fusion reduced graph node count and serialized model size.
- The smaller fused graph did not demonstrate lower inference latency in the controlled measurements.

One important takeaway:

**A smaller computation graph does not automatically mean faster inference. Performance needs to be measured on the actual workload and hardware.**

## Limitations

This was an exploratory experiment, not a comprehensive performance study.

- Only one embedding model was evaluated.
- The test input set contained four texts.
- Retrieval consistency was checked on a four-item corpus.
- Only CPU inference was evaluated.
- Benchmark rounds were performed in one process.
- Timing variability was substantial.
- Independent process-level benchmark validation remains future work.
- Results are specific to the tested software and hardware environment.

## Repository Contents

```text
transformers-onnx-export-benchmark/
│
├── README.md
└── notebook.ipynb
```

The notebook contains the export workflow, validation steps, benchmark measurements, graph inspection, and generated experiment reports.

## References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Transformers Exporters Documentation](https://huggingface.co/docs/transformers/main/en/exporters)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Model: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

*An independent learning and engineering experiment. The results should not be interpreted as a general performance comparison between PyTorch and ONNX Runtime.*
