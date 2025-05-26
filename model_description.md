# State Space Models

The increasing demand for architectures that handle long-range dependencies efficiently has driven a wave of innovation beyond Transformer-based models. Among the most promising approaches are State Space Models (SSMs), a class of architectures derived from classical control theory and linear dynamical systems. SSMs aim to bridge the gap between sequence modeling efficiency and expressive capacity by leveraging structured mathematical formulations that support fast convolutional operations and long memory retention.

This report explores the foundational SSMs and their structured variants, including S4, S4 Low-Rank, S4 Diagonal, S6, Mamba, and GatedMLP.

## S4

The foundational State Space Sequence model (S4) is derived from the continuous-time linear state space system:

$$
\begin{cases}
x'(t) & = Ax(t) + Bu(t)\\
y(t) & = Cx(t) + Du(t)\\
\end{cases}
$$

Here, $u(t)\in\mathbb{R}$ denotes the input signal, $x(t)\in\mathbb{R}^N$ represents the latent state, and $A, B, C, D$ are learnable parameters of the model. In practice, $D$ is often assumed to be zero and is therefore omitted.

This system is discretized for implementation as:

$$
\begin{cases}
x_k & = \bar{A}x_{k-1} + \bar{B}u_k\\
y_k & = \bar{C}x_k\\
\end{cases}
$$

where $\bar{A}, \bar{B}, \bar{C}$ denote the discretized forms of the corresponding continuous-time matrices.

While S4 is theoretically expressive, it suffers in practice from issues such as poor gradient scaling with respect to sequence length. To mitigate this, the matrix $A$ is typically initialized following the HiPPO (High-Order Polynomial Projection Operator) scheme:

$$
A_{ij} = -\begin{cases}
\sqrt{2i+1} \cdot \sqrt{2j +1} & if \quad i>j\\
i + 1 & if \quad i = j\\
0 & if \quad i<j
\end{cases}
$$

This initialization ensures stable dynamics and efficient representation of recent inputs while maintaining the potential for long-range information retention.

### S4 Low-Rank

The S4 Low-Rank variant generalizes the transition matrix $A$ to a diagonal plus low-rank structure:
$$
A = \Lambda + PQ^T
$$

where $\Lambda$ is a diagonal matrix and $PQ^T$ is a low-rank correction term. This formulation removes the HiPPO constraint, allowing $A$ to be fully learned during training.

The reduced structural rigidity yields greater flexibility and improved optimization while maintaining sufficient expressive power. The model strikes a balance between structure and learnability, enabling more efficient training and inference compared to the original S4.

### S4 Diagonal

S4 Diagonal further simplifies the architecture by restricting the transition matrix $A$ to be strictly diagonal, discarding both HiPPO and low-rank components.

This simplification significantly enhances computational and memory efficiency due to trivial parallelizability. Although its representational capacity is reduced, the model can still capture long-range dependencies when paired with expressive readout layers or embedding mechanisms. It is particularly well-suited to scenarios requiring lightweight and scalable models.

## S6

Selective State Space Models (S6) extend traditional SSMs by incorporating input-dependent selection mechanisms. This modification transforms SSMs from time-invariant to time-varying systems, wherein matrices $B$ and $C$ gain an additional length dimension, allowing them to adapt dynamically over the input sequence.

This architecture allows S6 to selectively retain or discard information across time steps based on input relevance. As a result, S6 achieves superior performance on tasks requiring selective memory, such as the selective copy task, outperforming earlier SSM variants including S4.

## Mamba

Mamba represents a modern, highly scalable SSM that builds upon S6. It integrates input-dependent gating to selectively control which state variables are updated at each time step—akin to attention mechanisms in Transformers, but with linear time and space complexity.

A key innovation in Mamba is the on-the-fly computation of convolution kernels via parameter-efficient neural filters and fused kernels. This enables the model to bypass full instantiation of the state space dynamics, significantly reducing memory overhead while preserving modeling power.

Mamba achieves competitive performance on challenging benchmarks, including language modeling and image generation, while maintaining the computational efficiency and scalability characteristic of SSMs.

## GatedMLP

Although not an SSM in the traditional sense, GatedMLP is often considered in the same context due to its sequence modeling capability without recurrence or attention. It consists of stacked multi-layer perceptrons with gating mechanisms, typically using element-wise multiplicative interactions, to model dependencies across tokens.

The typical form of a GatedMLP block is:

$$
y = (Wx)\odot\sigma(Vx)
$$

where $\odot$ denotes element-wise multiplication, and $\sigma$ is an activation function. While GatedMLPs lack explicit memory or recurrence, their gating allows for some level of sequential interaction when used in conjunction with positional embeddings or convolutional structures.

Despite their architectural simplicity and lack of explicit memory mechanisms, GatedMLPs can serve as efficient baselines. However, their performance generally lags behind that of structured SSMs.