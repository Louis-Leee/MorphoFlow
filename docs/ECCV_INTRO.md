% \begin{figure*}
%     \centering
%     \includegraphics[width=1.0\linewidth]{src/figure/rss_idea_new.pdf}
%     \caption{Overview of proposed framework. \textbf{Left (Training):} We collect a large multi-task dexterous in-hand manipulation dataset in simulation to pretrain a generative model that can generate diverse actions conditioned on the current state. The pretrained generative model can produce useful actions including rotation, translation, and more intricated behaviors.  \textbf{Right (Inference):} During inference, we can project dangerous motion produced by teleoperation or policy back to a high-likelihood action with guided sampling. This makes \mname{} capable of assisting a coarse high-level policy to perform complex object manipulations.}
%     \label{fig:enter-label}
% \end{figure*}

\section{Introduction}
Dexterous manipulation with multi-fingered hands represents a cornerstone of general-purpose robotics, offering the potential for human-level versatility in interacting with the physical world. However, equipping robots with this capability remains a formidable challenge. The difficulty stems not only from the high-dimensional state and action spaces inherent to dexterous manipulators but, more critically, from the immense diversity of robotic hardware. As the field expands, models must increasingly adapt to cross-embodiment settings where hands differ significantly in kinematic chain structures, degrees of freedom (DoF), and physical dimensions. Existing learning-based approaches typically grapple with this ``morphology gap'' using two different strategies. Robot-centric methods~\cite{wan2023unidexgrasp++, dfc, jiang2021graspTTA, huang2025fungrasp, liu2024realdex} often learn policies directly in the joint space of a specific hand. While efficient for that particular embodiment, these representations are implicitly tied to the robot's unique morphology, resulting in poor transferability and requiring expensive retraining for any new hardware. Conversely, object-centric or contact-based methods~\cite{shao2020unigrasp, xu2024manifoundation, xu2023unidexgrasp, li2023gendexgrasp, varley2015generating, attarian2023geometry, fang2025anydexgrasp, zhao2024graingrasp} focus on generating contact maps or grasp points on the object surface. Although these features are naturally morphology-agnostic, translating them back into feasible control actions for a specific hand typically requires complex, computation-heavy post-optimization or inverse kinematics solvers, which can be brittle and prone to failure when the desired contact is kinematically unreachable. 

Despite these advances, a fundamental limitation persists: most current cross-embodiment methods struggle to align distinct kinematic structures into a compatible action space. They often require the grasping model to simultaneously learn the complex physics of stable interaction and the kinematic constraints of every robot in the training set.~\cite{wei2024dro} attempts to address this by modeling the point-to-point map between the robot hand and the object with a suitable
initial configuration, making the model vulnerable when the starting state is infeasible.~\cite{fei2025tro} leverages a graph-based diffusion model to jointly encode hand morphology and object geometry, but the entanglement requires massive object-hand training data, which is difficult to scale up and limits generalization to unseen hands.
 Consequently, these models rarely achieve true zero-shot generalization: they typically fail when presented with a novel hand structure that was not explicitly included in the grasping training data. \LC{}{add a reference here}

To address these limitations, we introduce \emph{\mnamefull{}}, a novel framework inspired by the philosophy of mechanical assembly~\cite{li2025garf, sun2025_rpf}. We reformulate the grasp pose estimation of a dexterous hand as the spatial "assembly" of its links. Central to this approach is the use of link poses as a pivotal intermediate representation. This representation acts as a decoupling bridge: it isolates the intrinsic kinematic feasibility of the hand from the stable grasp geometry of the object. Consequently, we can learn a generalized policy that synthesizes valid grasps by searching this intermediate valid poses, avoiding overfitting to specific hand constraints.

We propose a two-stage paradigm to realize this vision. First, we introduce
Cross-morphology Pretraining, where we learn a unified morphology manifold. Instead
of relying on hand-specific joint angles, we learn to predict relative $SE(3)$
link poses from arbitrary joint configurations. This phase effectively maps disparate
kinematic structures ---regardless of their DoF or chain hierarchy--- into a shared
representation of feasible relative spatial relationships. Second, we formulate the
grasp synthesis problem as conditional generation with flow matching. In this stage, the model learns
to locate the object-specific sub-manifold corresponding to stable grasps within
the unified morphology manifold. This disentanglement is the key to
our model's generalization capabilities. Unlike prior works that require end-to-end
training on every target embodiment, \mnamefull{} can be trained on a partial
set of hands for the grasping task and yet achieve zero-shot generalization to
entirely novel hands. As long as a hand's kinematic structure was seen during the
morphology pretraining phase, which requires no object-grasp data, our model can
synthesize stable grasps for it without any additional fine-tuning. Meanwhile, we propose a physics-guided inference-time scaling strategy to further enhance the grasp quality. By converting the ODE process in flow matching to an SDE with a stochastic path, we can explore diverse grasp candidates and leverage a physics-based scoring function to select the best one. This refinement step significantly improves grasp stability and contact quality without retraining. In summary, our main contributions are as follows:
\begin{itemize}
    \item We propose MorphoFlow, a two-stage generative framework for cross-embodiment dexterous grasp synthesis that decouples hand kinematic feasibility from object-conditioned grasp reasoning. This formulation yields efficient grasp generation and consistent behavior across heterogeneous hand embodiments.

    \item \LC{}{first sentence seems to belong to previous point instead} We cast grasping as searching an object-specific stable sub-manifold in a shared morphology manifold, enabling generalization across heterogeneous hands and diverse objects. Meanwhile, the proposed physics-guided inference-time search can adaptively refine the synthesized grasps for better stability and contact quality without retraining the model.

    \item Extensive experiments demonstrate
        strong performance and transferability across embodiments in both simulation and real-world across objects with diverse geometry.
\end{itemize}