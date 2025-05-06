## Aim of the Project
This is an extended work on the original paper titled "Deep Reinforcement Learning for Optimal Portfolio Allocation: A ComparativeStudy with Mean-Variance Optimization", where we added trading cost to both, MVO and DRL and then compared the results that if DRL still outperform MVO or not.

## RL Model Used
We used Baseline3 model and trained with exact same parameters given in the original paper, except reducing the timesteps to 200k from 7.5M.
Data is used from Bloomberg and not yfinance for more accuracy.
