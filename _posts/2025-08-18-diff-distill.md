---
layout: distill
title: A Unified Framework for Diffusion Distillation 
description: In this blog post, we introduce a set of notations that can be well adapted to recent works on one-step or few-step diffusion models.
tags: generative-models diffusion flows
giscus_comments: true
date: 2025-08-21
featured: true

authors:
  - name: Yuxiang Fu
    url: "https://felix-yuxiang.github.io/"
    affiliations:
      name: UBC
  - name: Qi Yan
    url: "https://qiyan98.github.io"
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
    - name: Shortcut Model
    - name: ReFlow
    - name: Inductive Moment Matching
    - name: Distribution Matching Distillation
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

Diffusion and flow-based models have taken over generative AI space, enabling unprecedented capabilities in videos, audios, and text generation. Nonetheless, there is a caveat - they are painfully **slow** during inference. Generating a single high-quality sample will require running through hundreds of denoising steps, which translate to high costs and long wait times. 

At its core, diffusion models (equivalently, flow matching models) operate by iteratively refining noisy data into high-quality outputs through a series of denoising steps. Similar to divide-and-conquer algorithms <d-footnote>Common ones like Mergesort, locating the median and Fast Fourior Transform.</d-footnote> , diffusion models first *divide* the difficult denoising task into subtasks and *conquer* one of these at a time during training. To obtain a sample, we make a sequence of recursive predictions which means we need to *conquer* the entire task end-to-end. 

This challenge has spurred research into acceleration strategies across multiple grandular levels, including hardware optimization (e.g., high-FLOPs GPUs), mixed-precision training, quantization (e.g., using bitsandbytes), and parameter-efficient fine-tuning (e.g., LoRA adapters). In this blog, we focus on an orthogonal approach, ODE distillation techniques, which minimize Number of Function Evaluations (NFEs) so that we can generate high-quality samples with as few denoising steps as possible.

Distillation, in general, is a technique that transfers knowledge from a complex, high-performance model (the *teacher*) to a more efficient, customized model (the *student*). Recent distillation methods have achieved remarkable reductions in sampling steps, from hundreds to just a few and even **one** step, while preserving the sample quality. This advancement paves the way for real-time applications and deployment in resource-constrained environments.


## Notation at a Glance

Let's denote the data distribution and the noise distribution by $$\mathbf{x}_0\sim p_{\text{data}}, \mathbf{x}_1\sim p_{\text{noise}}$$ respectively, according to the original flow matching <d-cite key="lipman_flow_2023"></d-cite> setup. The target is to reconstruct the marginal flow path with high precision as $$\mathbf{x}_t\sim p_t, t\in[0,1]$$, and we denote a conditional flow path by $$\mathbf{x}_t\sim p_t(\cdot \vert \mathbf{x}_0).$$<d-footnote>In pratice, the most common one is the Gaussian conditional probability path. This is because it induces a Gaussian conditional vector field with analytical form. Checkout the detail in the table.</d-footnote>

Most of the conditional flow paths are designed as the linear interpolation between noise and data for simplicity, and we can express sampling from a marginal path 
$$\mathbf{x}_t = \alpha(t)\mathbf{x}_0 + \beta(t)\mathbf{x}_1$$ where $\alpha(t), \beta(t)$ are predefined schedules. For every datapoint $\mathbf{x}_0\in \mathbb{R}^d$, let $$v(\mathbf{x}_t, t\vert\mathbf{x}_0)=\mathbb{E}_{p_t(v_t | \mathbf{x}_0)}[v_t]$$ denote a conditional vector field so that the corresponding ODE yields the conditional probability path above,

$$
\require{physics}
\dv{\mathbf{x}_t}{t}=v(\mathbf{x}_t, t\vert\mathbf{x}_0),\quad \mathbf{x}_0\sim p_{\text{data}} 
$$

We provide some popular instances <d-footnote>Note we ignore the diffusion models with SDE formulation like DDPM since we concentrate on ODE distillation in this blog.</d-footnote> of these schedules in the table below. 

| Method | Probability Path $p_t$ | Vector Field $v(\mathbf{x}_t, t\vert\mathbf{x}_0)$ |
|--------|---------------------------|------------------------------|
| Gaussian |$\mathcal{N}(\alpha(t)\mathbf{x}_0,\beta^2(t)I_d)$ | $\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) \mathbf{x}_0 + \frac{\dot{\beta}_t}{\beta_t}\mathbf{x}_1$| 
| FM <d-cite key="lipman_flow_2023"></d-cite>| $\mathcal{N}(\mathbf{x}; t\mathbf{x}_1, (1-t+\sigma t)^2)$ | $\frac{\mathbf{x}_1 - (1-\sigma)\mathbf{x}_t}{1-\sigma+\sigma t}$ |
| iCFM <d-cite key="liu2022flow"></d-cite>| $\mathcal{N}( t\mathbf{x}_1 + (1-t)\mathbf{x}_0, \sigma^2)$ | $\mathbf{x}_1 - \mathbf{x}_0$ |
| OT-CFM <d-cite key="tong2023improving"></d-cite>| $q(z) = \pi(\mathbf{x}_0, \mathbf{x}_1)$ | $\mathbf{x}_1 - \mathbf{x}_0$ |
| VP-SI <d-cite key="albergo2023stochastic"></d-cite>| $\mathcal{N}( \cos(\pi t/2)\mathbf{x}_0 + \sin(\pi t/2)\mathbf{x}_1, \sigma^2)$ | $\frac{\pi}{2}(\cos(\pi t/2)\mathbf{x}_1 - \sin(\pi t/2)\mathbf{x}_0)$ |

The simplest form of conditional flow path is $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$ with the corresponding default conditional velocity field OT target $v(\mathbf{x}_t, t \vert \mathbf{x}_0)=\mathbb{E}[\dot{\mathbf{x}}_t\vert \mathbf{x}_0]=\mathbf{x}_1- \mathbf{x}_0.$

Borrowed from this [slide](https://rectifiedflow.github.io/assets/slides/icml_07_distillation.pdf) at this year ICML, the objective of ODE distillation have been categorized into three cases, forward loss, backward loss and tri-consistency loss. 




<span style="color: blue; font-weight: bold;">Training</span>: minimizing the conditional FM loss is equivalent to minimize the marginal FM loss, so the optimization problem becomes

$$
\arg\min_\theta\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} 
\left[ w(t) \left\| v_\theta(\mathbf{x}_t, t) - v(\mathbf{x}_t, t | \mathbf{x}_0) \right\|_2^2 \right]
$$

(explain w(t) !!!) Optimization problem

<span style="color: orange; font-weight: bold;">Sampling</span>: Solve $$\dfrac{d}{dt}\mathbf{x}_t=v_\theta(\mathbf{x}_t, t)$$ from the initial condition $$\mathbf{x}_1\sim p_{\text{noise}}$$ Use any ODE solver to take a couple of hundreds discrete steps. (iterative refinements)


## ODE Distillation methods


### MeanFlow

>Default conditional flow path and default conditional velocity field OT target

We define our **average velocity field** as 
$$
u(\mathbf{x}_t, t, s) \triangleq \frac{1}{t - s} \int_s^t v(\mathbf{x}_\tau, \tau) d\tau
$$

Differentiate both sides w.r.t. $t$ and consider that $s$ is independent of $t$ we obtain 

$$
\require{physics}
v(\mathbf{x}_t, t)=u +(t-s)\dv{u}{t}
$$

where we compute the total derivative of $u$ w.r.t. $t$. 
Expand this we obtain

$u_\text{tgt}=v - (t-s)(v\partial_{\mathbf{x}_t}u + \partial_t u)$  

**Training:**

$$
\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t, s} 
\left[ w(t) \left\| u_\theta(\mathbf{x}_t, t, s) - u_\text{tgt}(\mathbf{x}_t, t, s | \mathbf{x}_0) \right\|_2^2 \right]

$$

where $u_\text{tgt}=v - (t-s)(v\partial_{\mathbf{x}_t}u_{\theta^-} + \partial_t u_{\theta^-})$ 

Total derivative of $u$ is derived via this expression: `dudt=jvp(u_theta, (xt, s, t), (v, 0, 1))`


**Sampling:**

$$
\mathbf{x}_s = \mathbf{x}_t - (t-s)u_\theta(\mathbf{x}_t, t, s)
$$


### CM

>Default conditional flow path and default conditional velocity field OT target

CMs train a neural network $f_\theta(\mathbf{x}_t, t)$ to map noisy inputs $\mathbf{x}_t$ directly to their corresponding clean samples $\mathbf{x}_0$. Consequently, $f_\theta(\mathbf{x}_t, t)$ must satisfy the **Boundary conditions**$f_\theta(\mathbf{x}_0, 0) = \mathbf{x}_0$, which is typically enforced by parameterizing 

$$f_\theta(\mathbf{x}_t, t) = c_{\text{skip}}(t)\mathbf{x}_t + c_{\text{out}}(t)F_\theta(\mathbf{x}_t, t), c_{\text{skip}}(0) = 1, c_{\text{out}}(0) = 0.$$


CMs are trained to have consistent outputs between adjacent timesteps. They can be trained from scratch or distilled from given diffusion or flow models. 

1. **Discretized CM**

- **Training:**
$$
\mathbb{E}_{\mathbf{x}_t, t} \left[ w(t) d\left(f_\theta(\mathbf{x}_t, t), f_{\theta^-}(\mathbf{x}_{t-\Delta t}, t - \Delta t)\right) \right],
$$

- **Sampling**: 

$$
\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_1, 1)
$$
where $\theta^-$ denotes $\text{stopgrad}(\theta)$, $w(t)$ is a weighting function, $\Delta t > 0$ is the distance between adjacent time steps, and $d(\cdot, \cdot)$ is a distance function. 

Common choices include
$\ell_2$ loss $d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2^2$, 
Pseudo-Huber loss $d(\mathbf{x}, \mathbf{y}) = \sqrt{||\mathbf{x} - \mathbf{y}||_2^2 + c^2} - c$ 
LPIPS loss. 


Discrete-time CMs are sensitive to the choice of $\Delta t$, and require manually designed annealing schedules The noisy sample $\mathbf{x}_{t-\Delta t}$ at the preceding timestep $t - \Delta t$ is often obtained from $\mathbf{x}_t$ by numerically solving the PF-ODE, which can cause additional discretization errors.


2. **Continuous CM**

When using $d(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2^2$ and taking the limit $\Delta t \to 0$, Song et al. show that the gradient with respect to $\theta$ converges to 
- **Training:** 
$$
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t} \left[ w(t) f_\theta^{\top}(\mathbf{x}_t, t) \frac{\text{d}f_{\theta^-}(\mathbf{x}_t, t)}{\text{d}t} \right] 
$$ where 

$$
\frac{\text{d}f_{\theta^-}(\mathbf{x}_t, t)}{\text{d}t} = \nabla_{\mathbf{x}_t} f_{\theta^-}(\mathbf{x}_t, t) \frac{\text{d}\mathbf{x}_t}{\text{d}t} + \partial_t f_{\theta^-}(\mathbf{x}_t, t)
$$
is the tangent of $f_{\theta^-}$ at $(\mathbf{x}_t, t)$ along the trajectory of the PF-ODE $\frac{\text{d}\mathbf{x}_t}{\text{d}t}$

- **Sampling:**

Same as the Discretized Version. 

### FACM 

>Default conditional flow path and default conditional velocity field OT target
> $f_\theta(\mathbf{x}_t, t) = \mathbf{x}_t + (1-t)F_\theta(\mathbf{x}_t, t)$

This special case of consistency function holds only if $\mathbf{x}_0\sim p_{\text{noise}},\mathbf{x}_1\sim p_{\text{data}}$ which is opposite of what we have defined in the Problem Setup. To align with our definition, this consistency function that we choose should be 


$f_\theta(\mathbf{x}_t, t) = \mathbf{x}_t - tF_\theta(\mathbf{x}_t, t)$

**Consistency property** requires the total derivative of the consistency function to be zero

$$
\dfrac{df_\theta(\mathbf{x}_t, t)}{dt} = 0
$$
Borrow from consistency function defined from **CM**, we derive that the neural network must satisfy

$$F_\theta(\mathbf{x}_t, t) = v - t\frac{dF_\theta(\mathbf{x}_t, t)}{dt}.$$
Notice this is equivalent to **MeanFlow** where $s=0$ . (!!! requires explanation) This means CM objective directly forces the network $F_\theta(\mathbf{x}_t, t)$ to learn the properties of an average velocity field, thus enabling the 1-step generation shortcut.


**Training:**
$c_{CM}=(t,1), c_{FM}=(t,t)$

![[Screenshot 2025-08-14 at 15.18.09.png]]
!!! **Rewrite** the algorithm FACM in our notation latex code

**Sampling:**
1-step CM is the same as CM

Multi-step sampling (NFE ≥ 2) follows a standard iterative refinement process.
Equally spaced time stamp $t_i=\frac{i-1}{N}, i\in[N]$ 

!!! Change to our notation
$$
\hat{\mathbf{x}}_1 = \mathbf{x}_{t_i} + (1 - t_i) F_\theta(\mathbf{x}_{t_i}, c_\text{CM}) 
$$
And we can continue by computing the next sample $\mathbf{x}_{t_{i+1}} = t_{i+1}\hat{\mathbf{x}}_1 + (1 - t_{i+1})\mathbf{x}_0,$


### AYF - Flow Maps

Let's define what is a flow map. Flow maps generalize diffusion, flow-based and consistency models within a single unified framework by training a neural network $f_\theta(\mathbf{x}_t, t, s)$ to map noisy inputs $\mathbf{x}_t$ directly to any point $\mathbf{x}_s$ along the PF-ODE in a single step. Unlike consistency models, which only perform well for single- or two-step generation but degrade in multi-step sampling, flow maps remain effective at all step counts.

Flow Maps are CMs when $s=0$

**General BC**
$f_\theta(\mathbf{x}_t, t, t) = \mathbf{x}_t$ for all $t$. So that we have

$f_\theta(\mathbf{x}_t, t, s) = c_{\text{skip}}(t, s)\mathbf{x}_t + c_{\text{out}}(t, s)\mathbf{F}_\theta(\mathbf{x}_t, t, s)$ where $c_{\text{skip}}(t, t) = 1$ and $c_{\text{out}}(t, t) = 0$ for all $t$. 

In this work, we set $c_{\text{skip}}(t, s) = 1$ and $c_{\text{out}}(t, s) = (s - t)$

Hence, we have $f_\theta(\mathbf{x}_t, t, s) = \mathbf{x}_t + (s-t)\mathbf{F}_\theta(\mathbf{x}_t, t, s)$

**Training:**

1. AYF-Eulerian Map Distillation
Let $f_\theta(\mathbf{x}_t, t, s)$ be the flow map. Consider the loss function defined between two adjacent starting timesteps $t$ and $t' = t + \epsilon(s - t)$ for a small $\epsilon > 0$,} 

$$\mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\|f_\theta(\mathbf{x}_t, t, s) - f_{\theta^-}(\mathbf{x}_{t'}, t', s)\|_2^2\right],$$ 
 where $\mathbf{x}_{t'}$ is obtained by applying a 1-step Euler solver to the PF-ODE from $t$ to $t'$. In the limit as $\epsilon \to 0$, the gradient of this objective with respect to $\theta$ converges to:
 $$ \nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w'(t, s)\text{sign}(t - s) \cdot \mathbf{f}_\theta^\top(\mathbf{x}_t, t, s) \cdot \frac{\text{d}f_{\theta^-}(\mathbf{x}_t, t, s)}{\text{d}t}\right],$$
where $w'(t, s) = w(t, s) \times |t - s|$.


2.  AYF-Lagrangian Map Distillation
Let $f_\theta(\mathbf{x}_t, t, s)$ be the flow map. Consider the loss function defined between two adjacent ending timesteps $s$ and $s' = s + \epsilon(t - s)$ for a small $\epsilon > 0$,}$$\mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\|f_\theta(\mathbf{x}_t, t, s) - ODE_{s' \to s}[f_{\theta^-}(\mathbf{x}_t, t, s')]\|_2^2\right],$$ where $ODE_{t \to s}(\mathbf{x})$ refers to running a 1-step Euler solver on the PF-ODE starting from $\mathbf{x}$ at timestep $t$ to timestep $s$. In the limit as $\epsilon \to 0$, the gradient of this objective with respect to $\theta$ converges to: $$\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w'(t, s)\text{sign}(s - t) \cdot \mathbf{f}_\theta^\top(\mathbf{x}_t, t, s) \cdot \left(\frac{\text{d}f_{\theta^-}(\mathbf{x}_t, t, s)}{\text{d}s} - \mathbf{v}_\phi(f_{\theta^-}(\mathbf{x}_t, t, s), s)\right)\right],$$where $w'(t, s) = w(t, s) \times |t - s|$.


**Connections:**
1. In AYF-EMD: the standard flow matching loss appears if $s\to t$
2. In AYF-EMD: this reduces to continuous CM when $s=0$
3. The gradient of MeanFlow objective matches the AYF-EMD objective using an Euler parametrization up to a constant

$$\mathcal{L}_{\text{MeanFlow}}(\theta) = \mathbb{E}_{\mathbf{x}_t, t, s}\left[\left\|\mathbf{F}_\theta(\mathbf{x}_t, t, s) - \left(\frac{\text{d}\mathbf{x}_t}{\text{d}t} - (t - s)\frac{\text{d}\mathbf{F}_{\theta^-}(\mathbf{x}_t, t, s)}{\text{d}t}\right)\right\|_2^2\right].$$
4. AYF-EMD objective and EMD loss in Flow Maps (no SG, and let $\epsilon\to 0$)
$$
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\left\|\partial_t f_\theta(\mathbf{x}_t, t, s) + \nabla_{\mathbf{x}} f_\theta(\mathbf{x}_t, t, s) \cdot \frac{\text{d}\mathbf{x}_t}{\text{d}t}\right\|_2^2\right] = \nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\left\|\frac{\text{d}f_\theta(\mathbf{x}_t, t, s)}{\text{d}t}\right\|_2^2\right].
$$
5. AYF-LMD objective and the LMD loss (let $\epsilon\to 0$)

$$
\nabla_\theta \mathbb{E}_{\mathbf{x}_t, t, s}\left[w(t, s)\left\|\partial_s f_\theta(\mathbf{x}_t, t, s) - \mathbf{v}_\phi(f_\theta(\mathbf{x}_t, t, s), s)\right\|_2^2\right]
$$

## Shortcut Models
They propose an objective combining flow matching and a self-consistency loss
$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_t, t, s}\left[\left\|\mathbf{F}_\theta(\mathbf{x}_t, t, t) - \frac{\text{d}\mathbf{x}_t}{\text{d}t}\right\|_2^2 + \left\|\mathbf{f}_\theta(\mathbf{x}_t, t, s) - \mathbf{f}_{\theta^-}\left(\mathbf{f}_{\theta^-}\left(\mathbf{x}_t, t, \frac{t + s}{2}\right), \frac{t + s}{2}, s\right)\right\|_2^2\right]
$$

## Inductive Moment Matching 

According to the notation defined in **AYF - Flow maps**, it uses an MMD loss to match distributions of $\mathbf{f}_\theta(\mathbf{x}_t, t, s)$ and $\mathbf{f}_{\theta^-}(\mathbf{x}_r,r,s)$ where $s< r< t$ .


## Distribution Matching Distillation


# Closing Thoughts
We have done flow matching ODE distillation on human motion trajectory (put the reference here). Compared to more common approaches such as adversarial distillation derived from GANs or Maximum Mean Discrepancy as used in [IMM](#inductive-moment-matching), IMLE is a relatively niche method that aligns two distributions directly from their samples.



Table incorporates the pros and cons in all dimensions for every methods
### References

ABC all the images that I scrape from the other papers