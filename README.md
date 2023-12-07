# CS 8803 Explainable AI Project - Prof. Sonia Chernova

## Motivation:

In real-world situations when RL models are deployed the other agents in the environment can learn to cheat and break the RL model or the maybe the RL model itself doesn't perform well in that environment. It can get difficult to understand the root cause in such situations, since in both the cases reward may go down eventually and it become difficult to distinguish between the two cases and point out the root cause of poor performance of the agent

## Objective and Methodology: 

This survey aims to investigate methods to explain the behavior of Reinforcement Learning (RL) agents, particularly focusing on distinguishing between two scenarios.

1. A poorly trained RL agent vs Computer
2. A Well performing RL agent vs Adversarial attacking computer

We want to assess the effectiveness of two XAI methods in aiding users to differentiate between the above to scenarios

1. Interestingness element
2. Reward Decomposition

We are interested in leveraging eXplainable AI (XAI) techniques to empower participants, who possess expertise in RL, to identify and interpret distinct patterns of struggle exhibited by RL agents.
By combining multimedia elements, including videos and images, with participant insights, we aim to understand valuable information on the utility of XAI for enhancing user understanding of RL agent struggles.

## Findings

The generated explanations show that the number of interestingness highlights/moments is consistently more for environments where the Opponent is making adversarial attacks. The interestingness explanations was able to highlight an avg of 40% of adversarial attacks.

![Image 1](findings/attack_highlights.png)
![Image 2](findings/interestingness_compare.png)

## References
1. [InterestingnessXRL](https://github.com/SRI-AIC/InterestingnessXRL)
2. [rl-policies-attacks-defenses](https://github.com/davide97l/rl-policies-attacks-defenses)


