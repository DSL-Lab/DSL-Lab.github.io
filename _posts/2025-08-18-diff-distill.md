---
layout: distill
title: A Unified Framework for Diffusion Distillation 
description: The explosive growth in one-step and few-step diffusion models has taken the field deep into the weeds of complex notations. In this blog, we cut through the confusion by proposing a coherent set of notations that reveal the connections among these methods.
tags: generative-models diffusion flows
giscus_comments: true
date: 2025-08-21
featured: true

authors:
  - name: Yuxiang Fu
    url: "https://felix-yuxiang.github.io/"
    affiliations:
      name: UBC

bibliography: 2025-08-18-diff-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Notation at a Glance 
  - name: ODE Distillation methods
  - subsections:
    - name: MeanFlow
    - name: Consistency Models
    - name: Flow Anchor Consistency Model
    - name: Align Your Flow
  - name: Connections
  - subsections:
    - name: Shortcut Models
    - name: ReFlow
    - name: Inductive Moment Matching
  - name: Closing Thoughts

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

Diffusion and flow-based models<d-cite key="ho2020denoising, lipman_flow_2023, albergo2023stochastic, liu2022flow"></d-cite> have taken over generative AI space, enabling unprecedented capabilities in videos, audios, and text generation. Nonetheless, there is a caveat - they are painfully **slow** during inference. Generating a single high-quality sample will require running through hundreds of denoising steps, which translate to high costs and long wait times. 

At its core, diffusion models (equivalently, flow matching models) operate by iteratively refining noisy data into high-quality outputs through a series of denoising steps. Similar to divide-and-conquer algorithms <d-footnote>Common ones like Mergesort, locating the median and Fast Fourior Transform.</d-footnote>, diffusion models first *divide* the difficult denoising task into subtasks and *conquer* one of these at a time during training. To obtain a sample, we make a sequence of recursive predictions which means we need to *conquer* the entire task end-to-end. 

This challenge has spurred research into acceleration strategies across multiple grandular levels, including hardware optimization, mixed precision training<d-cite key="micikevicius2017mixed"></d-cite>, [quantization](https://github.com/bitsandbytes-foundation/bitsandbytes), and parameter-efficient fine-tuning<d-cite key="hu2021lora"></d-cite>. In this blog, we focus on an orthogonal approach, **ODE distillation**, which minimize Number of Function Evaluations (NFEs) so that we can generate high-quality samples with as few denoising steps as possible.

Distillation, in general, is a technique that transfers knowledge from a complex, high-performance model (the *teacher*) to a more efficient, customized model (the *student*). Recent distillation methods have achieved remarkable reductions in sampling steps, from hundreds to just a few and even **one** step, while preserving the sample quality. This advancement paves the way for real-time applications and deployment in resource-constrained environments.


## Notation at a Glance
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/diff-distill/teaser_probpath_velocity_field.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
From left to right:<d-cite key="lipman2024flowmatchingguidecode"></d-cite>conditional and marginal probability paths, conditional and marginal velocity fields. The velocity field induces a flow that dictates its instanenous movement across all points in space.
</div>

The modern approachs of generative modelling consist of picking some samples from a base distribution $$\mathbf{x}_1\sim p_{\text{noise}}$$, typically an isotropic Gaussian, and learning a map such that $$\mathbf{x}_0\sim p_{\text{data}}$$. The connection between these two distributions can be expressed by establishing an initial value problem controlled by the **velocity field** $v(\mathbf{x}_t, t)$,

$$
\require{physics}
\dv{\psi_t(\mathbf{x}_t)}{t}=v(\psi_t(\mathbf{x}_t), t),\quad\psi_0(\mathbf{x}_0)=\mathbf{x}_0,\quad \mathbf{x}_0\sim p_{\text{data}} \tag{1}
$$

where the **flow** $\psi_t:\mathbb{R}^d\times[0,1]\to \mathbb{R}^d$ is a diffeomorphic map with $$\psi_t(\mathbf{x}_t)$$ defined as the solution to the above ODE. If the flow satisfies the push-forward equation<d-footnote>This is also known as the change of variable equation: $[\phi_t]_\# p_0(x) = p_0(\phi_t^{-1}(x)) \det \left[ \frac{\partial \phi_t^{-1}}{\partial x}(x) \right].$</d-footnote> $$p_t=[\psi_t]_\#p_0$$, we say a **probability path** $$(p_t)_{t\in[0,1]}$$ is generated from the vector field. The goal of flow matching<d-cite key="lipman_flow_2023"></d-cite> is to find a velocity field $$v_\theta(\mathbf{x}_t, t)$$ so that it transforms $$\mathbf{x}_1\sim p_{\text{noise}}$$ to $$\mathbf{x}_0\sim p_{\text{data}}$$ when integrated. In order to receive supervision at each time step, one must predefine a condition probability path $$p_t(\cdot \vert \mathbf{x}_0)$$<d-footnote>In pratice, the most common one is the Gaussian conditional probability path. This arises from a Gaussian conditional vector field, whose analytical form can be derived from the continuity equation. $$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v) = 0$$ See the table for details.</d-footnote> associated with its velocity field. For every datapoint $$\mathbf{x}_0\in \mathbb{R}^d$$, let $$v(\mathbf{x}_t, t\vert\mathbf{x}_0)=\mathbb{E}_{p_t(v_t \vert \mathbf{x}_0)}[v_t]$$ denote a conditional velocity field so that the corresponding ODE (1) yields the conditional flow. 

Most of the conditional probability paths are designed as the **differentiable** interpolation between noise and data for simplicity, and we can express sampling from a marginal path 
$$\mathbf{x}_t = \alpha(t)\mathbf{x}_0 + \beta(t)\mathbf{x}_1$$ where $$\alpha(t), \beta(t)$$ are predefined schedules. <d-footnote>The stochastic interpolant paper defines this probability path that summarizes all diffusion models, with several assumptions. Here, we use a simpler interpolant for clean illustration.</d-footnote>



We provide some popular instances <d-footnote>We ignore the diffusion models with SDE formulation like DDPM<d-cite key="ho2020denoising"></d-cite> or ScoreSDE<d-cite key="song2020score"></d-cite> on purpose since we concentrate on ODE distillation in this blog.</d-footnote> of these schedules in the table below. 

| Method | Probability Path $p_t$ | Vector Field $u(\mathbf{x}_t, t\vert\mathbf{x}_0)$ |
|--------|---------------------------|------------------------------|
| Gaussian |$$\mathcal{N}(\alpha(t)\mathbf{x}_0,\beta^2(t)I_d)$$ | $$\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) \mathbf{x}_0 + \frac{\dot{\beta}_t}{\beta_t}\mathbf{x}_1$$| 
| FM <d-cite key="lipman_flow_2023"></d-cite>| $$\mathcal{N}(t\mathbf{x}_1, (1-t+\sigma t)^2I_d)$$ | $$\frac{\mathbf{x}_1 - (1-\sigma)\mathbf{x}_t}{1-\sigma+\sigma t}$$ |
| iCFM <d-cite key="liu2022flow"></d-cite>| $$\mathcal{N}( t\mathbf{x}_1 + (1-t)\mathbf{x}_0, \sigma^2I_d)$$ | $$\mathbf{x}_1 - \mathbf{x}_0$$ |
| OT-CFM <d-cite key="tong2023improving"></d-cite>| Same prob. path above with $$q(z) = \pi(\mathbf{x}_0, \mathbf{x}_1)$$ | $$\mathbf{x}_1 - \mathbf{x}_0$$ |
| VP-SI <d-cite key="albergo2023stochastic"></d-cite>| $$\mathcal{N}( \cos(\pi t/2)\mathbf{x}_0 + \sin(\pi t/2)\mathbf{x}_1, \sigma^2I_d)$$ | $$\frac{\pi}{2}(\cos(\pi t/2)\mathbf{x}_1 - \sin(\pi t/2)\mathbf{x}_0)$$ |

The simplest form of conditional probability path is $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$ with the corresponding default conditional velocity field OT target $$v(\mathbf{x}_t, t \vert \mathbf{x}_0)=\mathbb{E}[\dot{\mathbf{x}}_t\vert \mathbf{x}_0]=\mathbf{x}_1- \mathbf{x}_0.$$

Borrowed from this [slide](https://rectifiedflow.github.io/assets/slides/icml_07_distillation.pdf) at ICML2025, the objective of ODE distillation have been categorized into three cases, i.e., (a) **forward loss**, (b) **backward loss** and (c) **tri-consistency loss**. 

!!! a video explaining these three losses



<span style="color: blue; font-weight: bold;">Training</span>: Since minimizing the conditional Flow Matching (FM) loss is equivalent to minimize the marginal FM loss<d-cite key="lipman_flow_2023"></d-cite>, the optimization problem becomes

$$
\arg\min_\theta\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} 
\left[ w(t) \left\| v_\theta(\mathbf{x}_t, t) - v(\mathbf{x}_t, t | \mathbf{x}_0) \right\|_2^2 \right]
$$
where $w(t)$ is a reweighting function.

<span style="color: orange; font-weight: bold;">Sampling</span>: Solve $$\require{physics} \dv{\mathbf{x}_t}{t}=v_\theta(\mathbf{x}_t, t)$$ from the initial condition $$\mathbf{x}_1\sim p_{\text{noise}}.$$ Typically, an Euler solver or another high-order ODE solver is employed, taking a few hundred discrete steps through iterative refinements.


## ODE Distillation methods
Before introducing ODE distillation methods, it is imperative to define a general continuous-time flow map $$f_{t\to s}(\mathbf{x}_t, t, s)$$<d-cite key="boffi2025build"></d-cite> where it maps any noisy input $$\mathbf{x}_t, t\in[0,1]$$ to any point $$\mathbf{x}_s, s\in[0,1]$$ on the ODE that describes the probability flow aformationed. This is a generalization of flow-based distillation and consistency models within a single unified framework. The flow map is well-defined only if its **boundary conditions** satisfy $$f_{t\to t}(\mathbf{x}_t, t, t) = \mathbf{x}_t$$ for all time steps. One popular way to meet the condition is to parameterize the model as $$ f_{t\to s}(\mathbf{x}_t, t, s)= c_{\text{skip}}(t, s)\mathbf{x}_t + c_{\text{out}}(t,s)F_{t\to s}(\mathbf{x}_t, t, s) $$ where $$c_{\text{skip}}(t, t) = 1$$ and $$c_{\text{out}}(t, t) = 0$$ for all $$t$$.

At its core, ODE distillation boils down to how to strategically construct the training objective of the flow map $$f_{t\to s}(\mathbf{x}_t, t, s)$$ so that it can be efficiently evaluated during sampling. In addition, we need orchestrate the schedule of $$(t,s)$$ pairs for better training dynamics.

### MeanFlow 
MeanFlow<d-cite key="geng2025mean"></d-cite> can be trained from scratch or distilled from a pretrained FM model. The conditional probability path is defined as the linear interpolation between noise and data $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$ with the corresponding default conditional velocity field OT target $$v(\mathbf{x}_t, t \vert \mathbf{x}_0)=\mathbf{x}_1- \mathbf{x}_0.$$ The main contribution consists of identifying and defining a **average velocity field** which coincides with our flow map as 

$$
F_{t\to s}(\mathbf{x}_t, t, s)=u(\mathbf{x}_t, t, s) \triangleq \frac{1}{t - s} \int_s^t v(\mathbf{x}_\tau, \tau) d\tau=\dfrac{f_{t\to s}(\mathbf{x}_t, t, s)-f_{t\to t}(\mathbf{x}_t, t, t))}{s-t}
$$

where $$c_{\text{out}}(t,s)=s-t$$. This is great since it attributes actual physical meaning to our flow map.

Differentiating both sides w.r.t. $t$ and consider the assumption that $s$ is independent of $t$, we obtain the MeanFlow identity<d-cite key="geng2025mean"></d-cite>

$$
\require{physics}
v(\mathbf{x}_t, t)=F_{t\to s}(\mathbf{x}_t, t, s) +(t-s)\dv{F_{t\to s}(\mathbf{x}_t, t, s)}{t}
$$

where we further compute the total derivative and derive the target $$F_{t\to s}^{\text{tgt}}(\mathbf{x}_t, t, s)$$.

<span style="color: blue; font-weight: bold;">Training</span>: Adapting to our flow map notation, the training objective turns to

$$
\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t, s} 
\left[ w(t) \left\| F^\theta_{t\to s}(\mathbf{x}_t, t, s) - F_{t\to s}^{\text{tgt}}(\mathbf{x}_t, t, s | \mathbf{x}_0) \right\|_2^2 \right]
$$


where $$F_{t\to s}^{\text{tgt}}(\mathbf{x}_t, t, s\vert\mathbf{x}_0)=v - (t-s)(v\partial_{\mathbf{x}_t}F^{\theta^-}_{t\to s}(\mathbf{x}_t, t, s) + \partial_t F^{\theta^-}_{t\to s}(\mathbf{x}_t, t, s))$$ and $$\theta^-$$ means `stopgrad()`. Note `stopgrad` aims to avoid high order gradient computation. There are a couple of choices for $$v$$, we can substitute it with $$F_{t\to t}(\mathbf{x}_t, t, t)$$ or $$v(\mathbf{x}_t, t \vert \mathbf{x}_0)=\mathbf{x}_1- \mathbf{x}_0.$$ Again, MeanFlow adopts the latter to reduce computation. 
<details>
<summary>Loss type</summary>
Type (b) backward loss
</details>
In practice, the total derivative of $$F_{t\to s}(\mathbf{x}_t, t, s)$$ and the evaluation can be done in a single function call: `f, dfdt=jvp(f_theta, (xt, s, t), (v, 0, 1))`. Despite `jvp` operation only introduces one extra backward pass, it still incurs instability and slos down training. Moreover, the `jvp` operation is currently incompatible with the latest attention architecture. SpiltMeanFlow<d-cite key="guo2025splitmeanflow"></d-cite> circumvents this issue by enforcing another consistency identity $$(t-s)F_{t\to s} = (t-r)F_{t\to r}+(r-s)F_{r\to s}$$ where $$s<r<t$$. This implies a discretized version of the MeanFlow objective which falls into loss type (c).


<span style="color: orange; font-weight: bold;">Sampling</span>:
Either one-step or multi-step sampling can be performed. It is intuitive to obtain the following expression by the definition of average velocity field

$$
\mathbf{x}_s = \mathbf{x}_t - (t-s)f^\theta_{t\to s}(\mathbf{x}_t, t, s).
$$

In particular, we achieve one-step inference by setting $t=1, s=0$ and sampling from $$\mathbf{x}_1\sim p_{\text{noise}}$$.


### Consistency Models 

Essentially, consistency models (CMs)<d-cite key="lu2024simplifying"></d-cite> are our flow map when $$s=0$$, i.e., $$f_{t\to 0}(\mathbf{x}_t, t, 0).$$

**Discretized CM**

CMs are trained to have consistent outputs between adjacent timesteps along the ODE trajectory. They can be trained from scratch by consistency training or distilled from given diffusion or flow models via consistency distillation like MeanFlow. 

- <span style="color: blue; font-weight: bold;">Training</span>: When expressed in our flow map notation, the objective becomes
 
$$
\mathbb{E}_{\mathbf{x}_t, t} \left[ w(t) d\left(f_{t \to 0}^\theta(\mathbf{x}_t, t,0), f_{t \to 0}^{\theta^-}(\mathbf{x}_{t-\Delta t}, t - \Delta t,0)\right) \right],
$$

where $$\theta^-$$ denotes $$\text{stopgrad}(\theta)$$, $$w(t)$$ is a weighting function, $$\Delta t > 0$$ is the distance between adjacent time steps, and $d(\cdot, \cdot)$ is a distance metric.<d-footnote>Common choices include $\ell_2$ loss $d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2^2$, pseudo-Huber loss $d(\mathbf{x}, \mathbf{y}) = \sqrt{||\mathbf{x} - \mathbf{y}||_2^2 + c^2} - c$ and Learned Perceptual Image Patch Similarity (LPIPS) loss. </d-footnote>

- <span style="color: orange; font-weight: bold;">Sampling</span>: 
It is natural to conduct one-step sampling with CM

$$
\hat{\mathbf{x}}_0 = f^{\theta}_{1\to 0}(\mathbf{x}_1, 1,0),
$$

while multi-step sampling is also possible since we can compute the next noisy output $$\mathbf{x}_{t-\Delta t}\sim p_{t-\Delta t}(\cdot\vert \mathbf{x}_0)$$ using the prescribed conditional probability path at our discretion. Discrete-time CMs depend heavily on the choice of $$\Delta t$$ and often require carefully designed annealing schedules. To obtain the noisy sample $$\mathbf{x}_{t-\Delta t}$$ at the previous step, one typically evolves backward $$\mathbf{x}_t$$ by numerically solving the ODE, which can introduce additional discretization errors.

**Continuous CM**

When using $$d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2^2$$ and taking the limit $\Delta t \to 0$, Song et al.<d-cite key="song2020score"></d-cite> show that the gradient with respect to $\theta$ converges to a new objective with no $$\Delta t$$ involved.
- <span style="color: blue; font-weight: bold;">Training</span>: In our notation, the objective is

$$
\require{physics}
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t} \left[ w(t) (f^\theta_{t\to 0})^{\top}(\mathbf{x}_t, t,0) \dv{f^{\theta^-}_{t\to 0}(\mathbf{x}_t, t,0)}{t} \right] 
$$ 

where $$ \require{physics} \dv{f^{\theta^-}_{t\to 0}(\mathbf{x}_t, t,0)}{t} = \nabla_{\mathbf{x}_t} f^{\theta^-}_{t\to 0}(\mathbf{x}_t, t,0) \dv{\mathbf{x}_t}{t} + \partial_t f^{\theta^-}_{t\to 0}(\mathbf{x}_t, t,0)$$ is the tangent of $f^{\theta^-}_{t\to 0}$ at $(\mathbf{x}_t, t)$ along the trajectory of the ODE defined (1). Consistency Trajectory Models<d-cite key="kim2023consistency"></d-cite> extend this objective so that the forward loss (type (a)) becomes globally optimized. In this context, their intuition is that $$f^\theta_{t \to s}(\mathbf{x}_t, t, s)\approx f^\theta_{r \to s}(\texttt{Solver}_{t\to r}(\mathbf{x}_t, t, r), r, s).$$ The composition order on the right-hand side depends on the assumption of the solver of the teacher model.

- <span style="color: orange; font-weight: bold;">Sampling</span>

Same as the Discretized Version. CTMs<d-cite key="kim2023consistency"></d-cite> introduce a new sampling method called $$\gamma$$-sampling which controls the noise level of diffusing the intermediate noisy sample according to the conditional probability path during multi-step sampling.

<details>
<summary>Loss type</summary>
Type (b) backward loss, while CTMs<d-cite key="kim2023consistency"></d-cite> optimize type (a) forward loss, both locally and globally.
</details>

### Flow Anchor Consistency Model 

Similar to MeanFlow preliminary, Flow Anchor Consistency Model (FACM)<d-cite key="peng2025flow"></d-cite> also adopts the linear conditional probability path $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$ with the corresponding default conditional velocity field OT target $$v(\mathbf{x}_t, t \vert \mathbf{x}_0)=\mathbf{x}_1- \mathbf{x}_0.$$ In our flow maps notation, FACM parameterizes the model as $$ f^\theta_{t\to s}(\mathbf{x}_t, t, 0)= \mathbf{x}_t - tF^\theta_{t\to s}(\mathbf{x}_t, t, 0) $$ where $$c_{\text{skip}}(t,s)=1$$ and $$c_{\text{out}}(t,s)=-t$$.

FACM imposes a **consistency property** which requires the total derivative of the consistency function to be zero 
$$
\require{physics}
\dv{t}f^\theta_{t \to 0}(\mathbf{x}, t, 0) = 0.
$$

By substituting the parameterization of FACM, we have

$$\require{physics}
F^\theta_{t\to 0}(\mathbf{x}_t, t, 0)=v(\mathbf{x}_t, t)-t\dv{F^\theta_{t\to 0}(\mathbf{x}_t, t, 0)}{t}.
$$

Notice this is equivalent to [MeanFlow](#meanflow) where $$s=0$$. This indicates CM objective directly forces the network $$F^\theta_{t\to 0}(\mathbf{x}_t, t, 0)$$ to learn the properties of an average velocity field heading towards the data distribution, thus enabling the 1-step generation shortcut.


<span style="color: blue; font-weight: bold;">Training</span>: FACM training alogrithm equipped with our flow map notation. Notice that $$d_1, d_2$$ are $\ell_2$ with cosine loss and norm $\ell_2$ loss respectively, plus reweighting. Interestingly, they separate the training of FM and CM on disentangled time intervals. When training with CM target, we let $$s=0, t\in[0,1]$$. On the other hand, we set $$t'=2-t, t'\in[1,2]$$ when training with FM anchors.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/diff-distill/facm_training.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<span style="color: orange; font-weight: bold;">Sampling</span>: Same as CM.
<details>
<summary>Loss type</summary>
Type (b) backward loss
</details>

### Align Your Flow

Our notation incorporates a small modification of the flow map introduced by Align Your Flow<d-cite key="sabour2025align"></d-cite>, where we indicate the direction of the distillation. Hence, we say that Align Your Flow (AYF) the continuous-time flow map $$f^{\text{AYF}}(\mathbf{x}_t, t, s)=f_{t\to s}(\mathbf{x}_t, t, s).$$ Specifically, AYF selects a tigher set of boundary conditions $$c_{\text{skip}}(t,s)=1$$ and $$c_{\text{out}}(t,s)=s-t$$.

<span style="color: blue; font-weight: bold;">Training</span>:
The first variant of the objective, called AYF-**Eulerian Map Distillation**, is compatible with both distillation and training from scratch. 

$$
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\text{sign}(t - s) \cdot (f^\theta_{t \to s})^\top(\mathbf{x}_t, t, s) \cdot \frac{\text{d}f^{\theta^-}_{t\to s}(\mathbf{x}_t, t, s)}{\text{d}t}\right]
$$

It is intriguing that this objective reduces to the [continuous CM](#consistency-models) objective when $$s=0$$, while transforming to original FM objective when $$s\to t$$. In addition, CTMs<d-cite key="kim2023consistency"></d-cite> uses a discrete consistency loss with a fixed discretized time schedule comparing to AYF-EMD objective.
Regarding the second variant, named AYF-**Lagrangian Map Distillation**, it is only applicable to distillation from a pretrained flow model $$F^\delta_{t \to t}(\mathbf{x}_t,t,t)$$. 

$$
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\text{sign}(s - t) \cdot (f^\theta_{t \to s})^\top \cdot \left(\frac{\text{d}f^{\theta^-}_{t\to s}}{\text{d}s} - F^\delta_{s \to s}((f_{\theta^-}(\mathbf{x}_t, t, s), s,s)\right)\right].
$$

<span style="color: orange; font-weight: bold;">Sampling</span>: Same as CM. A combination of $$\gamma$$-sampling and classifier-free guidance.

The formulation of these objectives is majorly built on the Flow Map Matching<d-cite key="boffi2025build"></d-cite>. Similar to the trick in training [Meanflow](#meanflow) and [CMs](#consistency-models), they add a `stopgrad` operator to the loss to make the objective practical. In their appendix, they provide a detailed proof of why these objectives are equivalent to the objectives in Flow Map Matching<d-cite key="boffi2025build"></d-cite>. 

<details>
<summary>Loss type</summary>
Type (b) backward loss for AYF-EMD, type (a) forward loss for AYF-LMD.
</details>

## Connections
Now it is time to connect the dots with some previous existing methods. Let's frame their objectives in our flow map notation and identify their loss types.

### Shortcut Models
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/diff-distill/shortcut_model.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The diagram of Shortcut Models<d-cite key="frans2024one"></d-cite>
</div>
In essence, Shortcut models<d-cite key="frans2024one"></d-cite> augment the standard flow matching objective with a self-consistency regularization term. This additional loss component ensures that the learned vector field satisfies a midpoint consistency property: the result of a single large integration step should match the composition of two smaller steps traversing the same portion of the ODE trajectory. 

<span style="color: blue; font-weight: bold;">Training</span>: In the training objective, we neglect the input arguments and focus on the core transition between time steps. Again, we elaborate it with our flow map notation.

$$
\mathbb{E}_{\mathbf{x}_t, t, s}\left[\left\|F^\theta_{t\to t} - \dfrac{\text{d}\mathbf{x}_t}{\text{d}t}\right\|_2^2 + \left\|f^\theta_{t\to s} - f^{\theta^-}_{\frac{t+s}{2}\to s}\circ f^{\theta^-}_{t \to \frac{t+s}{2}}\right\|_2^2\right]
$$

where we adopt the same flow map conditions based on [AYF](#align-your-flow).


<span style="color: orange; font-weight: bold;">Sampling</span>: Same with MeanFlow yet with specific shortcut lengths.
<details>
<summary>Loss type</summary>
Type (c) tri-consistency loss
</details>

### ReFlow
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/diff-distill/rectifiedflow.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The diagram of rectified flow and ReFlow process<d-cite key="liu2022flow"></d-cite>
</div>
Unlike most ODE distillation methods that learn to jump from $$t\to s$$ according to our defined flow map $$f_{t\to s}(\mathbf{x}_t, t, s)$$, ReFlow<d-cite key="liu2022flow"></d-cite> takes a different approach by establishing new noise-data couplings so that the new model will generate straighter trajectories.<d-footnote>In the rectified flow paper<d-cite key="liu2022flow"></d-cite>, the straightness of any continuously differentiable process $$Z=\{Z_t\}$$ can be measured by $$S(Z)=\int_0^1\mathbb{E}\|(Z_1-Z_0)-\dot{Z}_t\|^2 dt$$ where $S(Z)=0$ implies the trajectories are perfectly straight.</d-footnote> In this case, this allows the ODE to be solved with fewer steps and larger step sizes. To some extent, this resembles the preconditioning from OT-CFM<d-cite key="tong2023improving"></d-cite> where they intentionally sample noise and data pairs jointly from an optimal transport map $$\pi(\mathbf{x}_0, \mathbf{x}_1)$$ instead of independent marginals.

<span style="color: blue; font-weight: bold;">Training</span>: Pair synthesized data from the pretrained model with the noise. Use this new coupling to train a student model with the standard FM objective.

<span style="color: orange; font-weight: bold;">Sampling</span>: Same as FMs.

### Inductive Moment Matching 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="blog/2025/diff-distill/IMM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The diagram of IMM<d-cite key="zhou2025inductive"></d-cite>
</div>
This recent method<d-cite key="zhou2025inductive"></d-cite> trains our flow map from scratch via matching the distributions of $$f^{\theta}_{t\to s}(\mathbf{x}_t, t, s)$$ and $$f^{\theta}_{r\to s}(\mathbf{x}_r, r, s)$$ where $$s<r<t$$. They adopt an Maximum Mean Discrepancy (MMD) loss to match the distributions.

<span style="color: blue; font-weight: bold;">Training</span>: In our flow map notation, the training objective becomes

$$
\mathbb{E}_{\mathbf{x}_t, t, s} \left[ w(t,s) \text{MMD}^2\left(f_{t \to s}(\mathbf{x}_t, t,s), f_{r \to s}(\mathbf{x}_{r}, r,s)\right) \right]
$$

where $$w(t,s)$$ is a weighting function.

<span style="color: orange; font-weight: bold;">Sampling</span>: Same spirit as [AYF](#align-your-flow).


## Closing Thoughts

The concept of a flow map offers a capable and unifying notation for summarizing the diverse landscape of diffusion distillation methods. Beyond these ODE distillation methods, an intriguing family of approaches pursues a more direct goal: training a one-step generator from the ground up by directly matching the data distribution from the teacher model.

The core question is: how can we best leverage a pre-trained teacher model to train a student that approximates the data distribution $$p_{\text{data}}$$ in a single shot? With access to the teacher's flow, several compelling strategies emerge. It becomes possible to directly match the velocity fields, minimize the KL divergence between the student and teacher output distributions<d-cite key="yin2024improved"></d-cite>, or align their respective score functions<d-cite key="wang2025uni"></d-cite>.

This leads to distinct techniques in practice. For example, adversarial distillation<d-cite key="yin2024improved, sabour2025align"></d-cite> employs a min-max objective to align the two distributions, while other methods like [IMM](#inductive-moment-matching) rely on statistical divergences like the Maximum Mean Discrepancy (MMD).

In our own work on human motion prediction<d-cite key="fu2025moflowonestep"></d-cite>, we explored this direction by using Implicit Maximum Likelihood Estimation (IMLE). IMLE is a potent, if less common, technique that aligns distributions based purely on their samples, offering a direct and elegant way to distill the teacher's knowledge without requiring an explicit density function or a discriminator.

Diffusion distillation is a dynamic field brimming with potential. The journey from a hundred steps to a single step is not just a technical challenge but a gateway to real-time, efficient generative AI applications.


