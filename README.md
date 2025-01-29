# REGENT T0

REGENT T0 is the REGENT team's efforts to create an open replication of the DeepSeek R1 training process. It is in an early state, and training runs are ongoing. Expect frequent changes to this repository.

## GRPO

T0 is currently using the TRL GRPO training code. It is inefficient, and in time will be replaced with a custom implementation, but is being used for testing of reward signals and dataset.

GRPO is similar to PPO, in that it computes loss using a non-differentiable reward function using the ratio between a calculated advantage and the probability ratio between two candidate policy models. GRPO eschews the common use of a value model to calculate per-token advantages in favor of a per-completion advantage, with each advantage being calculated as a function of the mean and standard deviation of rewards in a given group of completions.

## Rewards

GRPO has proven extremely capable in outcome supervision of large language models, and has recently been the subject of a great deal of attention following the release of DeepSeek R1.

DeepSeek R1 is unique from other reinforcement learning schemes, in that it places no upper bound on the length of responses and does not care much about their structure. The model can produce as much or as little text as necessary, as long as it provides a correct answer in a specific tag. This gives rise to complex behaviors, including "reasoning" and other System-2 thinking behaviors. As such, reproductions of this method are in high demand.

DeepSeek R1 uses a combination of LeetCode problems and mathematics exam problems as verifiable domains for GRPO. This project seeks to escalate in terms of complexity, from logic puzzles onto college-level mathematics and programming challenges. The phases are described below:

1. **Logic Puzzles** - In this phase, a small model will learn to generate reasoning traces in order to solve a diverse set of logic puzzles, requiring inductive reasoning and trial-and-error. Here, the "a-ha!" moment of a model independently learning backtracking should be apparent.
2. **Programming Challenges** - In this phase, a small-to-medium sized model will learn to generate reasoning traces in order to pass test cases and compiler warnings on a variety of problems sourced from competitive programming platforms such as LeetCode, expanding logical and programming abilities.
3. **Mathematics Challenges** - In this phase, a combination of mathematics exam questions and competitive mathematics questions will be posed to a medium-to-large model, which has already been trained using the previous two phases. It will be evaluated purely on the correct answer. Strong generalization should be visible in this system, as well as heightened creativity and other known effects of verifiable-rewards RL in models such as these.

## Training and Goals

Training is currently underway on a cluster of 8 cloud-sourced H100 GPUs. Completion time on small models (2B-7B) is around 24-36 hours, which is a solid turnaround time for iteration on reward function design. Larger models may require more tuning of reward functions to maximize utility of compute budget, as well as multi-machine parallelism and other complexities. As such, our current goal is to produce a 2B-7B model demonstrating heightened performance on logic, math, and programming puzzles compared to the base model, proving the viability of further research in this direction. Additionally, models will be evaluated for creative writing skills and conversational coherence, including their viability for hybrid RAG system such as the REGENT Architecture.