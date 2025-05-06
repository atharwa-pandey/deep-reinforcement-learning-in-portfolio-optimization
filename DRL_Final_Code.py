import numpy as np
import pandas as pd
import os
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# -----------------------------
# 1. Data Acquisition
# -----------------------------
def fetch_data():
    # tickers = ['XLF', 'XLE', 'XLY', 'XLC', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'XLP']
    # index_ticker = '^GSPC'
    # vix_ticker = '^VIX'
    # all_tickers = tickers + [index_ticker, vix_ticker]
    #
    # dfs = {}
    # for t in all_tickers:
    #     df = yf.download(t, start="2006-01-01", end="2022-01-01", progress=False)
    #     if df.empty or 'Close' not in df.columns:
    #         continue
    #     df = df[['Close']].ffill()
    #     dfs[t] = df
    # data = pd.concat(dfs.values(), axis=1)
    # data.columns = list(dfs.keys())
    # data.sort_index(inplace=True)
    # data.index = pd.to_datetime(data.index)
    # return data
    data = pd.read_csv('indices.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    print(data.columns.tolist())
    return data


def prepare_features(price_df):
    asset_cols = [col for col in price_df.columns if col not in ['SPX', 'VIX']]
    asset_prices = price_df[asset_cols]
    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    sp500 = price_df['SPX']
    sp_returns = np.log(sp500 / sp500.shift(1))
    vol20 = sp_returns.rolling(window=20).std()
    vol60 = sp_returns.rolling(window=60).std()
    vol_ratio = (vol20 / vol60).replace([np.inf, -np.inf], 0).fillna(0)
    vix = price_df['VIX']
    scaler = StandardScaler()
    vix_norm = pd.Series(scaler.fit_transform(vix.values.reshape(-1, 1)).flatten(), index=vix.index)
    return asset_returns, vol_ratio, vix_norm


# -----------------------------
# 2. DRL Environment
# -----------------------------
class PortfolioEnv(gym.Env):
    def __init__(self, returns, vol_ratio, vix, lookback=60, trading_cost=0.001):
        super(PortfolioEnv, self).__init__()
        self.returns = returns
        self.vol_ratio = vol_ratio
        self.vix = vix
        self.lookback = lookback
        self.trading_cost = trading_cost
        self.n_assets = returns.shape[1]
        self.A = 0.0
        self.B = 0.0
        self.eta = 1.0 / 252.0  # decay rate η
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.total_tc = 0.0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.n_assets + 2, lookback), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.lookback
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.total_tc = 0.0
        # Start tracking weights; record initial weight allocation.
        self.weights_history = [self.prev_weights.copy()]
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.returns.iloc[self.current_step - self.lookback:self.current_step]
        returns_obs = window.T.values
        vol_obs = self.vol_ratio.iloc[self.current_step - self.lookback:self.current_step].values.reshape(1, -1)
        vix_obs = self.vix.iloc[self.current_step - self.lookback:self.current_step].values.reshape(1, -1)
        obs = np.vstack([returns_obs, vol_obs, vix_obs])
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones(self.n_assets)
        new_weights = action / np.sum(action)
        tc = self.trading_cost * np.sum(np.abs(new_weights - self.prev_weights))
        self.total_tc += tc
        todays_log_ret = self.returns.iloc[self.current_step].fillna(0).values
        todays_simple = np.expm1(todays_log_ret)
        # 1) portfolio simple return
        port_simple = np.dot(new_weights, todays_simple)
        # 2) net simple return after cost
        R_t = port_simple - tc
        # 3) Differential Sharpe update
        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B
        denom = (self.B - self.A ** 2) ** 1.5
        if denom != 0:
            d_sharpe = (self.B * delta_A - 0.5 * self.A * delta_B) / denom
        else:
            d_sharpe = 0.0

        # 4) EMA updates
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        reward = d_sharpe
        # port_ret = np.dot(new_weights, todays_ret)
        # log_return_after_trading_cost = port_ret + np.log(1 - tc)
        # Update portfolio weights and record them.
        self.prev_weights = new_weights
        self.weights_history.append(new_weights.copy())
        self.current_step += 1
        done = self.current_step >= len(self.returns)
        info = {"total_tc": self.total_tc, "log_return": np.log(1+R_t)}
        return self._get_obs(), reward, done, False, info,


# -----------------------------
# 3. DRL Training and Evaluation
# -----------------------------
def train_drl_on_data(train_returns, vol_ratio, vix, lookback=60, trading_cost=0.001, timesteps=200000, n_seeds=5):
    print(f"Starting training on the drl_data")
    agents = []
    val_scores = []
    global train_returns_val, vol_ratio_val, vix_val
    for seed in range(n_seeds):
        print(f"Seed: {seed}")
        env_fn = lambda: PortfolioEnv(train_returns, vol_ratio, vix, lookback, trading_cost)
        env = DummyVecEnv([env_fn for _ in range(10)])
        model = PPO("MlpPolicy", env, verbose=0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    seed=seed,  n_steps=756, batch_size=1260, n_epochs=16, gamma=0.9, gae_lambda=0.9, clip_range=0.25,
                    learning_rate=lambda f: 3e-4*f)
        model.learn(total_timesteps=timesteps)
        agents.append(model)
        val_score, _, _, _ = evaluate_drl(model, train_returns_val, vol_ratio_val, vix_val, lookback, trading_cost)
        val_scores.append(val_score)
    best_idx = np.argmax(val_scores)
    best_agent = agents[best_idx]
    return best_agent, val_scores[best_idx]


def evaluate_drl(agent, test_returns, vol_ratio, vix, lookback=60, trading_cost=0.001):
    env = PortfolioEnv(test_returns, vol_ratio, vix, lookback, trading_cost)
    obs, _ = env.reset()
    cum_log_ret = 0.0
    equity_curve = []
    dates = test_returns.index[lookback:]
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        cum_log_ret += info["log_return"]
        equity_curve.append(np.exp(cum_log_ret) - 1)  # Equity expressed as portfolio value - 1
        if done:
            break
    cum_return = np.exp(cum_log_ret) - 1
    # Create a DataFrame of portfolio weights. The weights_history has one extra record (initial weights),
    # so we discard the first record to align with the equity curve dates.
    weight_df = pd.DataFrame(env.weights_history[1:], columns=test_returns.columns, index=dates)
    return cum_return, info["total_tc"], pd.Series(equity_curve, index=dates), weight_df


# -----------------------------
# 4. Performance Metrics
# -----------------------------
def compute_annual_return(curve):
    """
    Computes the annualized return given an equity curve.
    Assumes the initial portfolio value is 1.
    """
    total_value = 1 + curve.iloc[-1]  # final portfolio value
    n_days = len(curve)
    # Annualize based on 252 trading days per year
    return total_value ** (252 / n_days) - 1


def compute_annual_sharpe(curve, rf=0.0):
    """
    Computes the annualized Sharpe ratio based on the equity curve.
    The curve is assumed to represent cumulative returns (portfolio value = 1 + cumulative return).
    """
    portfolio_values = 1 + curve
    daily_returns = portfolio_values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() - rf) / daily_returns.std() * np.sqrt(252)


# -----------------------------
# 5. DRL-Only Rolling Backtest
# -----------------------------
def rolling_backtest(full_returns, vol_ratio, vix, lookback=60, trading_cost=0.001, n_seeds=5):
    full_returns.index = pd.to_datetime(full_returns.index)
    vol_ratio.index = full_returns.index
    vix.index = full_returns.index

    results = []
    drl_curves = {}
    drl_weights = {}  # Dictionary to store the asset weights over time

    for year in range(2012, 2025):
        print(f"Starting rolling_backtest for the year {year}")
        train_period = full_returns[(full_returns.index.year >= (year - 6)) & (full_returns.index.year <= (year - 2))]
        val_period = full_returns[full_returns.index.year == (year - 1)]
        # test_period = full_returns[full_returns.index.year == year]
        # if len(train_period) == 0 or len(val_period) == 0 or len(test_period) == 0:
        #     continue
        prev_year = year - 1
        prev_period = full_returns[full_returns.index.year == prev_year]
        if len(prev_period) < lookback: # not enough history to seed the lookback window
            continue
        prev_lookback = prev_period.iloc[-lookback:]

        test_year_data = full_returns[full_returns.index.year == year]
        if len(train_period) == 0 or len(val_period) == 0 or len(test_year_data) == 0:
            continue

        train_vol = vol_ratio.loc[train_period.index]
        val_vol = vol_ratio.loc[val_period.index]
        test_vol = pd.concat([vol_ratio.loc[prev_lookback.index],vol_ratio.loc[test_year_data.index]])
        train_vix = vix.loc[train_period.index]
        val_vix = vix.loc[val_period.index]
        test_vix = pd.concat([vix.loc[prev_lookback.index], vix.loc[test_year_data.index]])

        global train_returns_val, vol_ratio_val, vix_val
        train_returns_val = val_period.copy()
        vol_ratio_val = val_vol.copy()
        vix_val = val_vix.copy()

        # model_path = f"ppo_best_agent_with_bloom_data_samehyper_{year}.zip"
        #model_path = f"ppo_best_agent_with_bloom_data_samehyper_till_last_year_new_reward_{year}.zip"
        #model_path = f"ppo_best_agent_with_bloom_data_samehyper_till_last_year_new_reward_without_tc_{year}.zip"
        #model_path = f"ppo_best_agent_with_bloom_data_samehyper_2million_till_last_year_new_reward_{year}.zip"
        model_path = f"best_model_final_{year}.zip"
        if os.path.exists(model_path):
            best_agent = PPO.load(model_path)
        else:
            best_agent, _ = train_drl_on_data(train_period, train_vol, train_vix, lookback, trading_cost,
                                              timesteps=200000, n_seeds=n_seeds)
            best_agent.save(model_path)
            print(f"Dumping the best model for the year {year}")

        test_returns = pd.concat([prev_lookback, test_year_data])
        drl_return, drl_tc, drl_curve, weight_df = evaluate_drl(best_agent, test_returns, test_vol, test_vix, lookback,
                                                                trading_cost)
        annual_ret = compute_annual_return(drl_curve)
        annual_sharpe = compute_annual_sharpe(drl_curve)

        results.append({
            "Year": year,
            "DRL Cumulative Return": drl_return,
            "DRL Trading Cost": drl_tc,
            "Annualized Return": annual_ret,
            "Annualized Sharpe": annual_sharpe
        })
        drl_curves[year] = drl_curve
        drl_weights[year] = weight_df

    results_df = pd.DataFrame(results)
    return results_df, drl_curves, drl_weights

def plot_returns_bar(df):
    # 2) Force Year to be categorical
    df["Year"] = df["Year"].astype(str)

    # 3) Compute mean
    mean_ret = df["Annualized Return"].mean()

    # 4) Plot
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    sns.barplot(
        x="Annualized Return",
        y="Year",
        data=df.sort_values("Year"),
        color="skyblue",
        edgecolor="none"
    )

    # 0% line
    plt.axvline(0, color="black", lw=1)
    # mean line
    plt.axvline(mean_ret, color="grey", linestyle="--", lw=1)

    # annotate “Mean” just above the top bar
    plt.text(
        mean_ret + 0.005,  # tiny offset right
        -0.3,  # above the first (top) bar
        "Mean",
        color="grey",
        va="center"
    )

    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel("Returns")
    plt.ylabel("Year")
    plt.title("Annual returns")
    plt.xlim(df["Annualized Return"].min() * 1.1,
             df["Annualized Return"].max() * 1.1)
    plt.tight_layout()
    plt.savefig("Annualized_returns.png")
    plt.show()

# -----------------------------
# 6. Main Function with Plots
# -----------------------------
def main():
    prices = fetch_data()
    asset_returns, vol_ratio, vix_norm = prepare_features(prices)
    results_df, drl_curves, drl_weights = rolling_backtest(asset_returns, vol_ratio, vix_norm, lookback=60,
                                                           trading_cost=0.001, n_seeds=5)
    print(results_df.to_string(index=False))

    # Plot equity curves for each year
    for year, curve in drl_curves.items():
        plt.figure(figsize=(10, 5))
        plt.plot(curve.index, 1 + curve, label=f"Equity Curve {year}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.title(f"DRL Equity Curve for Year {year}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Equity_curve_{year}.png")
        plt.show()

    # Combined equity curve plot
    plt.figure(figsize=(12, 6))
    for year, curve in drl_curves.items():
        plt.plot(curve.index, 1 + curve, label=str(year))
        new_curve = 1 + curve
        new_curve.to_csv(f"equity_curve_{year}.csv")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("DRL Equity Curves by Year")
    plt.legend()
    plt.grid(True)
    plt.savefig("Combined_equity_curve.png")
    plt.show()

    # Plot asset weights over time for each year
    for year, weight_df in drl_weights.items():
        plt.figure(figsize=(10, 6))
        for asset in weight_df.columns:
            plt.plot(weight_df.index, weight_df[asset], label=asset)
        plt.xlabel("Date")
        plt.ylabel("Asset Weight")
        plt.title(f"Asset Weights Over Time for Year {year}")
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True)
        plt.savefig(f"Weights_{year}.png")
        plt.show()

    #print(results_df.columns.to_list())
    plot_returns_bar(results_df)
    results_df.to_csv("final_result_summary.csv")

if __name__ == "__main__":
    main()
