"""
CEU (Causal Explanation-based Unlearning) Framework
--------------------------------------------------

这是一个基于因果解释的机器学习遗忘框架，实现了"先解释，后遗忘"的新范式。

模块:
- cci: 因果贡献识别器 (Causal Contribution Identifier)
- ura: 遗忘驱动的排序调整器 (Unlearning-driven Rank Adjuster)
- ceu_framework: CEU框架主类
- run_unlearning_experiment: 遗忘实验运行脚本
- simple_unlearning_demo: 简单演示脚本

主要特性:
1. 高效性: 毫秒级响应时间
2. 局部性: 仅影响请求用户，不影响他人
3. 可解释性: 遗忘过程本身是透明的
"""