---
layout: distill
title: A Unified Framework for Diffusion Distillation 
description: In this blog post, we introduce a set of notations that can be well adapted to recent works on one-step or few-step diffusion models.
tags: metrics video generative-models
giscus_comments: true
date: 2025-08-18
featured: true

authors:
  - name: Yuxiang Fu
    url: "https://felix-yuxiang.github.io/"
    affiliations:
      name: UBC
  - name: Qi Yan
    url: "https://qiyan98.github.io
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
  - name: Fréchet Video Motion Distance (FVMD)
  - subsections:
    - name: Video Key Points Tracking
    - name: Key Points Velocity and Acceleration Fields
    - name: Motion Feature
    - name: Visualizations
    - name: Fréchet Video Motion Distance
  - name: Experiments
  - subsections:
    - name: Sanity Check
    - name: Sensitivity Analysis
    - name: Quantitative Results
    - name: Human Study
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