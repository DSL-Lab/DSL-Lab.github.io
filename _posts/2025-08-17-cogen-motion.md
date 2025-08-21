---
layout: distill
title:  Conditional Generative Models for Motion Prediction
description: In this blog post, we discuss good engineering practices and the lessons learned—sometimes the hard way—from building conditional generative models (in particular, flow matching) for motion prediction problems.
tags: motion-prediction trajectory generative-models
giscus_comments: true
date: 2025-08-17
featured: true

authors:
  - name: Qi Yan
    url: "https://qiyan98.github.io/"
    affiliations:
      name: UBC
  - name: Yuxiang Fu
    url: "https://felix-yuxiang.github.io/"
    affiliations:
      name: UBC

bibliography: 2025-08-17-cogen-motion.bib

  
# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Challenges of Multi-Modal Prediction
  - name: Engineering Practices and Lessons
    subsections:
      - name: Data-Space Predictive Learning Objectives
      - name: Joint Multi-Modal Learning Losses
  - name: Exploring Inference Acceleration
  - name: Summary

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---



## Introduction

Needless to say, diffusion-based generative models (equivalently, flow matching models) are amazing inventions. They have shown great capacity to produce high-quality images, videos, audios and more, whether being unconditional on the benchmark datasets and conditioned on certain content in the wild.
In this blog, we discuss a relatively less explored application of **generative models for motion prediction**, which is a fundamental problem in many applications such as autonomous driving and robotics.

In a nutshell, motion prediction is the task of predicting the future trajectories of objects given their past trajectories, plus all sorts of available context information such as surrounding objects and high-fidelity maps.    
The said pipeline implemented by neural networks is simply:
```
Past Trajectory + Context Information ---> Neural Network ---> Future Trajectory
```

To produce meaningful future trajectories, we condition the generative models on the past trajectory and the context information.
Borrowed from our paper <d-cite key="fu2025moflow"></d-cite>, the pipeline looks like this:
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/cogen-motion/noise_to_traj_moflow.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The pipeline of motion prediction using conditional (denoising) generative models <d-cite key="fu2025moflow"></d-cite>.
</div>

The early datasets on human motion prediction mostly do not come with heavy context information, such as the well-known ETH-UCY and the SDD datasets (see more for summarization at <d-cite key="ivanovic2023trajdata"></d-cite>), which the above figure accurately depicts.
However, modern industry-standard datasets such as the Waymo Open Motion Dataset <d-cite key="ettinger2021large"></d-cite> and the Argoverse series datasets <d-cite key="wilson2023argoverse, chang2019argoverse"></d-cite> come with much richer context information, such as high-fidelity maps and other rich context information, which need more compute to process. No matter how complex the context information is, the generative model must be guided to **produce spatially and temporally coherent trajectories consistent with the past**. 

## Challenges of Multi-Modal Prediction

Motion *prediction*, as the name suggests, is inherently a forecasting task. For each input in a dataset, only one realization of the future motion is recorded, even though multiple plausible outcomes often exist. This mismatch between the inherently **multi-modal** nature of future motion and the **single ground-truth** annotation poses a core challenge for evaluation.  

In practice, standard metrics require models to output multiple trajectories, which are then compared against the observed ground truth. For example, **ADE (Average Displacement Error)** and **FDE (Final Displacement Error)** measure trajectory errors, and the minimum ADE/FDE across predictions is typically reported. This setup implicitly encourages models to produce diverse hypotheses, but only rewards the one closest to the recorded future.  Datasets such as Waymo Open Motion <d-cite key="ettinger2021large"></d-cite> and Argoverse <d-cite key="wilson2023argoverse, chang2019argoverse"></d-cite> extend evaluation with metrics targeting uncertainty calibration. For instance, Waymo’s **mAP** rewards models that assign higher confidence to trajectories closer to the ground truth. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/cogen-motion/vehicle_1_trajflow.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multi-modal trajectory forecasting made by TrajFlow <d-cite key="yan2025trajflow"></d-cite> on the Waymo Open Motion Dataset <d-cite key="ettinger2021large"></d-cite>. Multiple predictions are visualized using different colors, while the single ground truth is shown in red. 
</div>

The strong dependency of current evaluation metrics on a single ground truth, assessed instance by instance, poses a particular challenge for generative models. Although the task inherently requires generating diverse trajectories, models are only rewarded when one of their outputs happens to align closely with the recorded ground truth.  

As a result, the powerful ability of generative models to produce diverse samples from noise <d-cite key="ho2020denoising, lipman2022flow"></d-cite> does not necessarily translate into better performance under current metrics. For example, MotionDiffuser <d-cite key="jiang2023motiondiffuser"></d-cite>, a diffusion-based model that generates one trajectory at a time, requires a complex post-processing pipeline—ranging from likelihood-based filtering to hand-crafted attractor/repeller cost functions and non-maximum suppression (NMS) for outlier removal—in order to achieve reasonably good results.  

## Engineering Practices and Lessons

Now let's dive into the technical side of the problem.
In the forward process of flow matching, we adopt a simple linear interpolation between the clean trajectories $$Y^1 \sim q$$, where $$q$$ is the data distribution, and pure Gaussian noise $$Y^0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$:

$$
Y^t = (1-t)Y^0 + tY^1 \qquad t \in [0, 1].
$$

The reverse process, which allows us to generate new samples, is governed by the ordinary differential equations (ODEs):

$$
\mathrm{d} Y^t = v_\theta(Y^t, t, C)\mathrm{d}t,
$$

where $$v_\theta$$ is the parametrized vector field approximating the straight flow $$U^t = Y^1 - Y^0$$. Here, $$C$$ denotes the aggregated contextual information of agents in a scene, including the past trajectory and any other available context information.

### Data-Space Predictive Learning Objectives

From an engineering standpoint, a somewhat **bitter lesson** we encountered is that **existing predictive learning objectives remain remarkably strong**. Despite the appeal of noise-prediction formulations (e.g., $\epsilon$-prediction introduced in DDPM <d-cite key="ho2020denoising"></d-cite> and later adopted in flow matching <d-cite key="lipman2022flow"></d-cite>), straightforward predictive objectives in the data space—such as direct $$\hat{x}_0$$ reconstruction in DDPM notation<d-footnote>
Note that we follow the flow matching notations in <d-cite key="lipman2022flow"></d-cite> to use $t=1$ as the data distribution and $t=0$ as the noise distribution, which is opposite to the original DDPM notations in <d-cite key="ho2020denoising"></d-cite>.</d-footnote>—consistently yields more stable convergence.


Concretely, by rearranging the original linear flow objective, we define a neural network

$$
D_\theta := Y^t + (1-t)v_\theta(Y^t, C, t),
$$

which is trained to recover the future trajectory $$Y^1$$ in the data space. The corresponding objective is:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{Y^t, Y^1 \sim q, \, t \sim \mathcal{U}[0,1]} \left[ \frac{\| D_{\theta}(Y^t, C, t) - Y^1 \|_2^2}{(1 - t)^2} \right].
$$

Our empirical observation is that data-space predictive learning objectives outperform denoising objectives. We argue that this is largely influenced by the current evaluation protocol, which heavily rewards model outputs that are close to the ground truth.  

During training, the original denoising target matches the vector field $Y^1 - Y^0$, defined as the difference between the data sample (future trajectory) and the noise sample (drawn from the noise distribution). Under the current proximity-based metrics, this objective is harder to optimize than the predictive objective because of the stochasticity introduced by $Y^0$, as the metrics do not adequately reward diverse forecasting. Moreover, during the sampling process, small errors in the vector field model $v_\theta$—measured with respect to the single ground-truth velocity field at intermediate time steps—can be amplified through subsequent iterative steps. Consequently, increasing inference-time compute may not necessarily improve results without incorporating regularization from the data-space loss <d-footnote>
Interestingly, in our experiments, we found that flow-matching ODEs—thanks to their less noisy inference process—usually perform more stably than diffusion-model SDEs, which is surprising. In image generation, as shown in SiT <d-cite key="ma2024sit"></d-cite>, ODE-based samplers are generally weaker than SDE-based samplers.
</d-footnote>.

### Joint Multi-Modal Learning Losses

Building on this, another key engineering practice was to introduce **joint multi-modal learning losses**. Our network $$D_\theta$$ generates $$K$$ scene-level correlated waypoint predictions $$\{S_i\}_{i=1}^K$$ along with classification logits $$\{\zeta_i\}_{i=1}^K$$<d-footnote>
Usually, different datasets have different conventions for what a proper $K$ should be. For example, $K=20$ is used for the ETH-UCY dataset, while $K=6$ is used for the Waymo Open Motion Dataset <d-cite key="ettinger2021large"></d-cite>.
</d-footnote>. This allows us to capture diverse futures in a single inference loop while still grounding learning in a predictive loss.
Such a principle of combined regression and classification losses to encourage trajectory multi-modality is ubiquitous in the motion prediction literature, as seen in MTR <d-cite key="shi2022motion"></d-cite>, UniAD <d-cite key="hu2023planning"></d-cite>, and QCNet <d-cite key="zhou2023query"></d-cite>, though these methods differ in other implementation details.
For simplicity, we omit the time-dependent weighting and define the multi-modal flow matching loss:

$$
\bar{\mathcal{L}}_{\text{FM}} = \mathbb{E}_{Y^t, Y^1 \sim q, \, t \sim \mathcal{U}[0,1]} \left[ \| S_{j^*} - Y^1 \|_2^2 + \text{CE}(\zeta_{1:K}, j^*) \right],
$$

where $$j^* = \arg\min_{j} \| S_j - Y^1 \|_2^2$$ indicates the closest waypoint to the ground-truth trajectory and $$\text{CE}(\cdot,\cdot)$$ denotes cross-entropy. On tasks where confidence calibration is important, such as those measured by the mAP metric in the Waymo Open Motion Dataset, we refer readers to our paper <d-cite key="yan2025trajflow"></d-cite> for further details on uncertainty calibration. 

We acknowledge that some prior works, such as MotionLM <d-cite key="seff2023motionlm"></d-cite> and MotionDiffuser <d-cite key="jiang2023motiondiffuser"></d-cite>, generate one trajectory at a time and have demonstrated strong performance. However, since these methods are not open-sourced, we are unable to conduct direct comparisons or measure their runtime efficiency. We conjecture that requiring multiple inference loops (tens to hundreds) is considerably slower than our one-step generator—particularly on smaller-scale datasets, where the one-step approach achieves comparable performance without significant degradation.  

## Exploring Inference Acceleration

To accelerate inference in flow-matching models, which typically require tens or even hundreds of iterations for ODE simulation, we adopt an underrated idea from the image generation literature: conditional **IMLE (implicit maximum likelihood estimation)** <d-cite key="li2018implicit, li2019diverse"></d-cite>. IMLE provides a way to distill an iterative generative model into a **one-step generator**.  

The IMLE family consists of generative models designed to produce diverse samples in a single forward pass, conceptually similar to the generator in GANs <d-cite key="goodfellow2020generative"></d-cite>. In our setting, we construct a conditional IMLE model that takes the same context $$C$$ as the teacher flow-matching model and learns to match the teacher’s motion prediction results directly in the data space.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/cogen-motion/imle_moflow.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of the IMLE distillation process in our work <d-cite key="fu2025moflow"></d-cite>.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/cogen-motion/IMLE_algorithm.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The IMLE distillation process is summarized in `Algorithm 1`. Lines 4–6 describe the standard ODE-based sampling of the teacher model, which produces $K$ correlated multi-modal trajectory predictions $$\hat{Y}^1_{1:K}$$ conditioned on the context $C$. A conditional IMLE generator $G_\phi$ then uses a noise vector $Z$ and context $C$ to generate $K$-component trajectories $\Gamma$, matching the shape of $$\hat{Y}^1_{1:K}$$.  

Unlike direct distillation, the conditional IMLE objective generates **more** samples than those available in the teacher’s dataset for the same context $C$. Specifically, $m$ i.i.d. samples are drawn from $G_\phi$, and the one closest to the teacher prediction $$\hat{Y}^1_{1:K}$$ is selected for loss computation. This nearest-neighbor matching ensures that the teacher model’s modes are faithfully captured.  

To preserve trajectory multi-modality, we employ the Chamfer distance <d-cite key="fan2017point"></d-cite> $d_{\text{Chamfer}}(\hat{Y}^1, \Gamma)$ as our loss function:  

$$
\mathcal{L}_{\text{IMLE}}(\hat{Y}^1_{1:K}, \Gamma) = \dfrac{1}{K} \left( \sum_{i=1}^K \min_j \|\hat{Y}^1_i - \Gamma^{(j)}\| + \sum_{j=1}^K \min_i \|\hat{Y}^1_i - \Gamma^{(j)}\| \right)
$$

where $\Gamma^{(i)} \in \mathbb{R}^{A \times 2T_f}$ is the $i$-th component of the IMLE-generated correlated trajectory.

Nonetheless, the acceleration of diffusion-based models—particularly through distillation—is evolving rapidly. Our work with IMLE is just one attempt in this direction, and we are actively exploring further improvements to extend its applicability to broader domains.


## Summary

We reviewed the challenges and engineering insights gained from developing conditional generative models for motion prediction, primarily drawing on our previous works <d-cite key="fu2025moflow, yan2025trajflow"></d-cite>. The task requires generating diverse trajectories, yet common evaluation metrics such as ADE and FDE primarily reward alignment with a single ground-truth trajectory.  

From these experiences, we identified two useful engineering practices:  
- Data-space predictive learning objectives outperform denoising-based approaches, leading to more stable convergence.  
- Joint multi-modal learning losses that integrate regression and classification more effectively capture trajectory diversity.  

In addition, we explored the IMLE distillation technique to accelerate inference by compressing iterative processes into a one-step generator, while preserving multi-modality through Chamfer distance losses.  
