## Reviewer #1

### 1. Paper Summary
The paper analyzes how student models in Knowledge Distillation (KD) learn features using the Interaction Tensor framework, finding that KD acts as a regularizer that prunes low-frequency, sample-specific features in favor of a compact set of reusable, high-frequency ones. The authors then observe that a model's own confusion matrix captures inter-class similarity structures analogous to a teacher's "dark knowledge." Building on this, they propose Confusion Distillation (CD), a teacher-free self-distillation method that uses the model's evolving confusion patterns as dynamic soft targets, outperforming existing self-distillation methods on CIFAR-100.

### 2. Paper Strengths
* The use of the Interaction Tensor framework to analyze KD at the feature level is a fresh and rigorous approach, going beyond the typical output-distribution analyses to provide concrete, quantitative insights into what students actually learn.
* The core idea — that a model's own confusion matrix encodes inter-class relationships similar to a teacher's dark knowledge — is elegant, well-reasoned, and backed by quantitative evidence (Pearson correlation ~0.87, cosine similarity ~0.78).
* Confusion Distillation requires no external teacher, no additional network, and no extra data, making it computationally efficient and easy to integrate into existing training pipelines.
* The authors test across multiple architectures (ResNet-18, 34, 50) and carefully tune hyperparameters, lending credibility to their results. The use of error bars (±) also reflects good experimental rigor.
* Rather than presenting analysis and method as separate contributions, the paper does a commendable job of letting the analytical findings directly motivate the proposed method, giving the work a coherent and satisfying narrative flow.

### 3. Paper Weaknesses
* Experiments are conducted exclusively on CIFAR-100. The absence of evaluations on larger, more challenging benchmarks like ImageNet significantly limits the generalizability of the claims, and the authors themselves acknowledge this as future work.
* The paper only compares CD against CS-KD and PS-KD, which are relatively older self-distillation methods. Comparisons against more recent and stronger self-distillation baselines would have made the performance gains more convincing.
* The accuracy gains over competing methods are quite small (often within ~0.5–1%), raising questions about whether the improvements are practically meaningful or largely within the noise margin, despite the use of error bars.
* While the Jaccard similarity (~0.41) and Pearson correlation (~0.87) suggest structural overlap between confusion ratios and teacher softmax outputs, the theoretical grounding for why confusion patterns approximate dark knowledge remains largely empirical and somewhat hand-wavy.
* The method requires careful tuning of the transition schedule and soft-hard loss ratio, and the paper shows that poor choices can significantly degrade performance. This adds practical complexity and reduces the method's plug-and-play appeal.
* The Interaction Tensor analysis relies on PCA projections and clustering across many models simultaneously, which may not scale well to larger architectures or datasets, limiting the broader applicability of the analytical framework itself.

---

## Reviewer #2

### 1. Paper Summary
This paper investigates the internal mechanisms of knowledge distillation (KD), focusing specifically on how student networks learn feature representations from teachers. The paper uses the Interaction Tensor framework to analyse baseline models, teacher models, and student models trained via KD, in order to find a connection. The conclusion is that Teacher models rely on fewer but more reusable features and student models trained with KD mimic the teacher’s feature usage pattern. Based on that, they propose a self-distillation model, where the model’s confusion matrix is used as a soft supervision signal.

### 2. Paper Strengths
* The authors offer a more interpretable view of how KD modifies feature utilization, which is a valuable contribution.
* They also conclude that KD suppresses low-frequency, sample-specific features. In addition, KD seems to promote reusable high-frequency features and student networks adopt teacher-like feature usage distributions. These are very valuable conclusions.
* Finally, they propose a self-distillation method that appears to work well in the presented cases.

### 3. Paper Weaknesses
* The only weaknesses I can find is that the study is based only on CIFAR 100 and only using the ResNet family. A more thorough ablation with other datasets and deep learning frameworks would be welcome to ensure the generalisation of the derived conclusions.

---

## Reviewer #3

### 1. Paper Summary
The paper provides a theoretical and empirical investigation into the mechanics of Knowledge Distillation (KD), specifically exploring how "dark knowledge" from a teacher model influences a student's feature acquisition. Using the Interaction Tensor framework, the authors analyze the frequency and reusability of learned features, discovering that effective KD acts as a regularizer that prunes sample-specific, low-frequency features in favor of highly reusable ones. Building on the observation that a dataset-level confusion matrix mirrors the structural information of a teacher’s soft targets, the authors propose Confusion Distillation (CD). CD is a teacher-free self-distillation method that uses a model's own evolving confusion patterns as dynamic soft targets, achieving competitive performance on ResNet architectures without the computational burden of a large teacher model.

### 2. Paper Strengths
* A major strength of this work is the transition from heuristic-based distillation to a formal feature-level analysis. By employing the Interaction Tensor framework, the authors provide a quantifiable explanation for why KD works, which is a significant contribution to the interpretability of model compression.
* The proposal of Confusion Distillation (CD) is highly innovative; it successfully bridges the gap between standard KD and self-distillation by identifying the "Dark Knowledge" inherent in the model’s own confusion patterns.
* The method is computationally efficient and demonstrates consistent performance gains over established self-distillation techniques like CS-KD and PS-KD, making it a practical solution for resource-constrained environments.

### 3. Paper Weaknesses
* The primary weakness lies in the diversity of the experimental validation. While the results on CIFAR-100 with ResNet architectures are promising, the paper lacks evaluation on large-scale datasets such as ImageNet-1K, which is the standard for proving the scalability of distillation methods.
* Additionally, the analysis of the Interaction Tensor is mathematically dense, and the paper would benefit from a clearer explanation of how these tensors are computed in a practical training loop for deeper networks.
* There is also a lack of comparison with modern feature-based distillation methods (e.g., FitNets or ReviewKD), which would help clarify whether the "dark knowledge" found in the confusion matrix is superior to direct feature-map alignment.

---

## Meta-Reviewer #1

### 1. Meta-Review
This paper provides an investigation into the internal mechanics of Knowledge Distillation (KD). Using the Interaction Tensor framework, the authors analyze how "dark knowledge" from a teacher model influences a student's feature acquisition. The authors propose Confusion Distillation (CD), a teacher-free self-distillation method that uses the model’s own evolving confusion matrix as a dynamic soft supervision signal, mimicking the structural information typically provided by a teacher's soft targets.

* The reviewers pointed out that the paper presents these main strengths: formal feature-level analysis that allows Interpretability and theoretical depth, an innovative self-distillation approach, a practical approach, and interesting conclusions.
* The reviewers pointed out that the paper presents these main weaknesses: limited experimental diversity (only one dataset and only an evaluated model – Resnet family), lack of evaluation on large-scale benchmarks, lack of clear and practical explanation of the theoretical concepts used, lack of comparison with baselines, and lack of discussion on topics such as class imbalance.