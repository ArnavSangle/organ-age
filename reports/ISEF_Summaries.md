# Organ-Age: ISEF Project Summaries

**Project Title:** Organ-Age: Multimodal Fusion of Transcriptomic and Radiological Signals for Organ-Resolved Biological Age Estimation

**Author:** Arnav Sangle, Ranchview High School, Irving TX

---

## Summary 1 — Highly Technical (AI Engineer / Research Scientist)

**Audience:** AI/ML engineer at a top research lab (e.g., Google Brain, DeepMind) with deep familiarity with contrastive learning, transformers, and multimodal fusion.

---

**Problem & Motivation**

Existing biological age predictors (epigenetic clocks, transcriptomic aging scores) collapse a heterogeneous, organ-specific process into a single scalar. This conflates systemic aging with organ-local signals and precludes clinical localization of accelerated aging. We address this by building a multi-modal, organ-conditioned age regressor that jointly models RNA-seq transcriptomics, chest X-ray structural information, and T1-weighted brain MRI — three modalities with disjoint biological scales and largely non-overlapping sample populations.

**Architecture**

*Modality-specific encoders.* A feedforward MLP encodes variance-stabilized DESeq2 RNA-seq counts (ComBat batch-corrected); fine-tuned ResNet-18 backbones (BiT initialization) handle 2-D chest radiographs and 3-D MRI slices (axial projections). All encoders project to a shared 256-D embedding space.

*Contrastive alignment (v3).* Projection heads trained with InfoNCE (cosine similarity, temperature τ learned) align inter-modality embeddings. The negative sampling strategy treats same-age-bin cross-modality pairs as soft positives, introducing a biological prior into the contrastive objective. Post-alignment, UMAP confirms collapse of modality-driven clustering in favor of age-gradient organization.

*Transformer fusion (v3.5 → v4.5).* Aligned embeddings are fused via a 4-layer TransformerEncoder (8 heads, GELU, d_model=256). The v4.5 CrossFusionV45 architecture concatenates a learned organ embedding token (Embedding(11, 64) → Linear(64→256)) with the fused embedding token as a 2-token sequence, enabling the attention mechanism to condition predictions on organ identity without separate organ-specific heads. The regression head outputs (μ, log σ²) trained under heteroscedastic Gaussian NLL loss with softplus-gated σ to prevent collapse.

*Post-hoc calibration.* Isotonic regression on a held-out validation split recalibrates μ toward the chronological age distribution; confidence intervals are generated via empirical quantile bootstrapping over calibration residuals.

*Attribution (v4.5).* Integrated Gradients are computed in the aligned latent space (256-D z_rna) per organ-conditioned prediction. A separate per-gene importance score is derived by projecting latent-space IG magnitudes back through the linear layers of the RNA encoder's bottleneck. SHAP TreeExplainer is applied to an XGBoost surrogate trained on encoder outputs as a secondary attribution method.

**Results**

- **MAE 9.3 yr** over 195,766 samples (all modalities combined, v3.5).
- MRI achieves 6.2 yr MAE with only 563 samples — evidence that structural brain information is highly age-informative and benefits substantially from alignment to a richer RNA/X-ray prior.
- Contrastive alignment reduces modality-separation entropy in latent space and cuts inter-modality fusion variance. Residual heteroscedasticity is lower post-alignment than in the naïve concatenation baseline.
- Organ-specific IG analyses reveal that attribution mass concentrates on a sparse subset of latent RNA dimensions (<15% of 256-D carry >80% of attribution for each organ), suggesting the encoder distills aging signals into a compact, reusable factor structure.

**Limitations & Open Problems**

The three datasets have no individual-level overlap; alignment thus operates on distributional rather than matched-sample statistics, limiting causal cross-modal claims. The current setup conflates epistemic and aleatoric uncertainty (the Gaussian NLL head captures aleatoric; epistemic estimation would require MC-Dropout or deep ensembles). Extension to a true CLIP-style pre-training with matched cross-modal subject samples would be a meaningful next step, as would adding proteomics (CPTAC) or longitudinal clinical records as fourth/fifth modalities.

---

## Summary 2 — Moderately Technical (Parent Judge with AI/ML Fundamentals)

**Audience:** Parent judge who has completed an introductory AI/ML course — familiar with terms like neural network, loss function, and feature importance, but not with contrastive learning or transformer architecture details.

---

**What problem does this project solve?**

Your biological age — how "old" your body actually is — doesn't have to match your birthday. Some people's hearts age faster than normal while their livers stay young; others show the opposite pattern. Most existing tools for measuring biological age produce just one number for the whole body, hiding these organ-by-organ differences. This project builds a system called **Organ-Age** that gives a separate biological age estimate for each major organ.

**What data does it use?**

Organ-Age uses three types of publicly available medical data, each measuring aging from a different angle:

1. **Gene expression (RNA-seq from GTEx):** Which genes are switched on or off in a given tissue? Aging changes gene activity in characteristic ways. This dataset covers ~7,400 samples from dozens of human tissue types.

2. **Chest X-rays (CheXpert):** The skeleton, heart silhouette, and soft tissue visible in a chest X-ray all change with age. This dataset has ~187,800 images.

3. **Brain MRI (IXI):** Brain volume, tissue contrast, and structural details in T1-weighted MRI scans correlate with age. This dataset provides ~563 scans.

**How does the AI work?**

Think of the project as a three-stage pipeline:

1. **Encoding** — Three separate neural networks each learn to compress their respective data type (genes, X-rays, MRI) into a compact 256-number summary vector (called an *embedding*). Each embedding captures aging-relevant patterns in its modality.

2. **Alignment** — A technique called *contrastive learning* trains the three networks to produce compatible embeddings. Without this step, the gene-based embeddings and image-based embeddings "speak different languages" and can't be compared. Contrastive learning encourages all three to speak the same language by pulling embeddings from similar biological contexts close together.

3. **Fusion and prediction** — A *transformer* (the same architecture powering large language models like ChatGPT) combines the aligned embeddings and produces a predicted biological age for each organ, along with an uncertainty estimate so the model can say "I'm less confident about this prediction."

**Why does contrastive learning matter?**

Without it, concatenating gene and image embeddings produced an MAE (average prediction error) higher than the aligned version. After alignment, the embeddings organize by age rather than by data type — meaning the fusion step can extract genuinely shared aging signals from both genes and images.

**What are the main results?**

- The combined model predicts biological age with a mean error of **~9.3 years** across nearly 196,000 samples.
- MRI predictions are even more accurate (6.2 yr error) despite having only 563 training scans, suggesting brain structural aging is strongly encoded.
- Different organs show different gap distributions — some age faster or slower than the chronological clock — consistent with what is known from biology (e.g., livers are sensitive to metabolic stress, brains to neurodegenerative processes).
- The system can also explain *which genes* drive the age prediction for each organ, giving researchers a list of molecular targets to investigate.

**Why does this matter?**

A tool that tells you which organ is aging unusually fast — and which genes may be responsible — could eventually help doctors detect early organ-specific disease risk years before symptoms appear, or help scientists understand why some organs age faster than others.

---

## Summary 3 — Non-Technical (Parent Judge with Software Background, No AI/ML)

**Audience:** Parent judge with general software development experience but no background in machine learning or biology.

---

**The Big Idea**

We all have a calendar age — how many years since we were born. But inside our bodies, different organs age at different speeds. A 50-year-old smoker may have the lungs of a 70-year-old and the heart of a 45-year-old at the same time. Knowing *which* organ is aging abnormally, and by how much, is a powerful piece of medical information — but no existing tool can measure this directly.

**Organ-Age** is a software system that estimates organ-specific biological age by analyzing three kinds of medical data simultaneously: gene activity data, chest X-ray images, and brain MRI scans.

**What the system does — in plain terms**

Think of it like a team of three specialists:

- A **genomics expert** looks at which genes are turned on or off in a tissue sample and notes which patterns look older or younger than average.
- A **radiologist** studies a chest X-ray and recognizes structural features — bone density, heart size, tissue texture — that change predictably with age.
- A **neuroimaging specialist** examines a brain scan and picks up on volume changes and tissue contrasts that signal aging.

Normally these three experts work in separate clinics and never compare notes. Organ-Age builds a unified system that gets all three to agree on a shared "language" for aging signals — then combines their assessments to produce a single, organ-by-organ aging report.

**How was it built?**

The system was built entirely in Python, using open-source machine learning libraries. Three separate neural networks (programs that learn patterns from data rather than following hard-coded rules) were trained — one for each data type. A technique called *contrastive learning* then taught the three networks to produce compatible outputs, so that "young liver gene expression" and "young liver-area on X-ray" end up described in the same mathematical terms. A final network (called a transformer, the same technology behind modern AI assistants) combines everything and outputs an age estimate for each organ, along with a confidence range.

**What were the results?**

The system was tested on a dataset of nearly 200,000 medical samples. On average it predicted biological age within about **9.3 years** of chronological age — a meaningful result given that biological age and calendar age are genuinely different quantities. Importantly, different organs did show different aging patterns, confirming the biological hypothesis that aging is not the same everywhere in the body.

The system also produces explanations: for each organ, it lists the specific genes whose activity levels most influenced the age prediction, giving scientists a concrete hypothesis to test in follow-up experiments.

**Why it matters**

Think of this as a proof of concept for a future diagnostic tool — a kind of "multi-organ biological age blood panel" that gives doctors organ-specific aging information rather than a single whole-body number. Earlier detection of which organ is aging abnormally could translate into earlier intervention, better health outcomes, and a richer understanding of how and why individual people age differently.
