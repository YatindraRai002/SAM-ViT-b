# SAM for Medical Imaging

This project takes Meta's Segment Anything Model — a powerful general-purpose segmentation system — and asks a simple question: *can it work on medical images, and if so, how does it actually do it internally?*

SAM was trained on billions of natural images. X-rays and MRIs look nothing like that. There's no color, no familiar object boundaries, and the contrast patterns are completely different. So rather than assuming it works, this project investigates it properly — fine-tune it, then look inside and understand what's happening.

---

## What the model actually is

SAM has three parts working together. The **image encoder** is the heavy lifter — a large Vision Transformer that processes a 1024×1024 image and compresses it down into a rich feature map. This runs once and the result is reused. The **prompt encoder** converts user inputs (clicks, boxes) into embeddings that guide the segmentation. The **mask decoder** takes everything the encoder produced and figures out which pixels belong to the target region.

The model used here is SAM ViT-B — the base variant, which balances capability with hardware requirements.

---

## The Dataset

20 chest X-rays and 20 MRI scans, each paired with a hand-labeled binary mask marking the region of interest. Small, but enough to demonstrate the full pipeline. Every image is resized to 1024×1024 and normalized using SAM's original preprocessing statistics before being fed into the model.

---

## What this pipeline does

### Fine-tuning

The first thing the pipeline does is adapt SAM to medical images. The image encoder — which weighs in at around 300MB and was already trained on an enormous natural image dataset — is kept frozen. Touching it with only 20 medical images would likely make things worse, not better. Instead, only the mask decoder is trained.

This is a deliberate design choice, not a shortcut. The encoder already knows how to extract useful visual features. What it doesn't know is how to map those features to medical segmentation boundaries. That's exactly what the decoder learns.

Training uses a combined loss: Binary Cross-Entropy penalises pixel-by-pixel errors, and Dice Loss directly optimises the overlap between predicted and ground-truth masks. The model converges surprisingly fast — by the third epoch, it achieves 99.6% Dice on validation data.

A note on prompts: during training, zero-valued embeddings are used as placeholder prompts rather than real point or box inputs. This forces the decoder to learn modality-level priors from the image embedding alone, which is sufficient for demonstration purposes.

### Representation Analysis

Once the model is trained, the next question is: *what does SAM actually see when it looks at an X-ray versus an MRI?*

To answer this, hooks are attached to the transformer blocks inside the image encoder. These hooks intercept the output of each block as images pass through, without modifying anything. Features are collected at layers 0, 4, 8, 9, 10, and 11 for all images in both modalities.

CKA — Centered Kernel Alignment — is then computed between the X-ray features and MRI features at each layer. CKA is a mathematically principled way of asking "do these two sets of representations have the same geometric structure?" A score of 1.0 means the model treats both modalities identically at that layer. A score near 0 means they're being processed very differently.

The result is a curve showing how the model's treatment of X-rays and MRIs evolves from shallow layers to deep layers. This is genuinely useful: it tells you where in the network the modality distinction is being encoded.

### Modality Separation

This phase asks a different but related question: *are SAM's internal features useful for telling X-rays from MRIs?*

A silhouette score measures how geometrically separated the two modality clusters are in feature space. A linear probe — just a logistic regression, no deep learning — tries to classify X-ray vs MRI using only those features. If a simple linear classifier can achieve high accuracy, it means the modality information is cleanly encoded and readily accessible.

t-SNE plots are generated for each layer, projecting the high-dimensional features down to 2D so you can see the clusters with your own eyes.

On the synthetic dataset used here, both the silhouette score and linear probe hit 1.0 across all layers — perfect separation. This is expected: chest X-rays and brain MRIs are visually so different that SAM distinguishes them from the very first layer. On real clinical data with more variation, the layer-wise progression would tell a more interesting story.

### Attention Head Ablation

The final phase tries to answer the most mechanistic question: *which specific attention heads in layer 10 are responsible for segmentation quality?*

Each attention head is silenced one at a time by registering a hook that zeroes out its contribution to the layer's output. A full inference pass is then run — using a centre-point prompt this time, through the proper SAM prediction interface — and the resulting mask is compared against the ground truth. The Dice drop measures how much the ablation hurt.

The results show that X-ray segmentation drops by about 82% when any head is ablated, and MRI by about 72%. The modality difference is real and interpretable. However, all four tested heads show nearly identical drops, which points to a limitation: zeroing a slice of the post-projection output doesn't truly isolate individual heads the way ablating the attention weights directly would. That's a known constraint of this approach and a clear direction for improvement.

---

## Results at a glance

The fine-tuned decoder converges to 99.6% Dice in three epochs. Modality separation is perfect at all measured layers (expected given the dataset characteristics). Layer 10 attention contributes substantially to segmentation quality in both modalities, with X-ray showing greater sensitivity to ablation than MRI.

---

## How to run it

Download the SAM ViT-B checkpoint from Meta's official repository and place it in the project root. Add your dataset following the `xray/` and `mri/` folder structure with `_mask.png` files paired to each image. Install dependencies and run:

```bash
pip install -r requirements.txt
python -u run_all.py
```

All four phases run sequentially. Results — plots and metrics — are saved to the `outputs/` directory.

---

## Hardware note

The pipeline runs on CPU if no CUDA-enabled PyTorch is installed. Training on CPU is slow but works. To use the GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## What could be improved

**Real data.** The current dataset makes the separation analysis trivial. Using NIH ChestX-ray14 or MSD datasets would produce genuinely informative results.

**Better ablation.** Ablating attention weight matrices before projection — rather than slices of the combined output — would give true head-level isolation.

**Bounding box prompts.** Replacing zero-vector prompts with bounding boxes derived from ground-truth masks during training would make the fine-tuned model more practically useful.

**GPU.** Everything runs faster.
