# Kalman filter pairs trading, explained

Code: [`kalman.py`](../src/qml/pairs/kalman.py), [`backtest.py`](../src/qml/pairs/backtest.py), [`run_kalman_pairs.py`](../experiments/run_kalman_pairs.py).

## 1. The problem in plain English

Coca-Cola (KO) and Pepsi (PEP) sell similar products, so their stock prices tend to rise and fall together. Sometimes KO jumps while PEP does not. **Pairs trading** bets the gap will close: sell the one that ran ahead, buy the other, and collect when they realign.

To judge whether the gap is unusually big you need to know how much PEP "should" move per KO move. That number is the **hedge ratio** $\beta$ (like a gear ratio). The trouble is that $\beta$ is not fixed; it drifts as the companies change. A **Kalman filter** is a recipe for tracking something hidden that changes slowly. Think of a GPS: it has a guess of where you are, predicts where you will be, then corrects the guess using each new reading, trusting the reading more or less depending on how noisy it is.

## 2. The equations, every symbol

| Symbol | Meaning |
|---|---|
| $y_t, x_t$ | log price of the stock we model (KO) and of the stock we hedge with (PEP) |
| $\beta_t$ | hedge ratio (hidden, drifting) |
| $\alpha_t$ | intercept: a constant gap between the two (hidden, drifting) |
| $R$ | noise: how far $y$ scatters around the line |
| $\delta$ | how fast the hidden state can drift |
| $P$ | the filter's uncertainty about $\beta, \alpha$ |
| $K$ | Kalman gain: how much of today's surprise to believe |
| $e_t$ | prediction error: actual $y$ minus what yesterday's estimates predicted |
| $z_t$ | the error in units of its expected size |

$$y_t = \beta_t x_t + \alpha_t + \varepsilon_t, \quad \varepsilon \sim N(0, R)$$

$$\theta_t = \theta_{t-1} + w_t, \quad w \sim N\left(0, \tfrac{\delta}{1-\delta} I\right)$$

$$K_t = \frac{P_{t|t-1} h_t}{h_t^\top P_{t|t-1} h_t + R}, \qquad \hat\theta_t = \hat\theta_{t|t-1} + K_t e_t$$

where $\theta = (\beta, \alpha)$ and $h_t = (x_t, 1)$.

**Knobs:**

- **$\delta$ up:** the filter believes $\beta$ moves fast, so it chases every wiggle. Then the leftover "gap" shrinks toward nothing and there is no signal. In my sweep, $\delta \ge 10^{-4}$ produced zero trades.
- **$\delta$ down:** $\beta$ is nearly fixed, like one regression over all history. Slow to react.
- **$R$ up:** today's data is noisy, so the filter trusts it less (small gain) and the z-score shrinks.
- **$R$ too small:** every day looks like a huge surprise. I fit $R$ by maximum likelihood on training data so $|z| > 2$ means roughly what it should.
- **Entry threshold up** (say 2.5): fewer, more extreme trades. **Exit threshold up:** quicker exits, smaller profits per trade, more trading cost.

## 3. Walkthrough of the code, in running order

1. `get_prices` downloads the pair; `analyse_pair` takes log prices and splits at 60%.
2. `engle_granger_pvalue` tests the training window for **cointegration** (do the two prices share a mean-reverting gap?). Small p-values (< 0.05) support pairs trading.
3. `sweep_delta` loops over candidate $\delta$ values. For each it calls `fit_obs_var` (find $R$ by maximum likelihood), `run_filter`, and `run_backtest` on training data, and reports the Sharpe ratio and trade count.
4. `choose_delta` picks the best Sharpe among candidates that traded at least 20 times (a Sharpe from 3 trades is luck).
5. `KalmanHedge.update` is the heart. For each day it: (a) adds drift uncertainty to $P$ (predict); (b) computes the error $e$ and its expected variance; (c) computes the gain $K$; (d) corrects the state and shrinks $P$ (update). It returns $\beta$, $\alpha$, $e$, and the expected std so $z = e/\text{std}$.
6. `rolling_ols` is the rival: a plain regression over the previous 60 days, shifted by a day so it too uses only the past.
7. `run_backtest` walks the days: it earns the return of yesterday's position, then looks at today's $z$. Enter short (long) the spread if $z > 2$ ($z < -2$); exit when $|z| < 0.5$; subtract 5 bps whenever the position changes. Positions are held with $\beta$ frozen at entry. Today's signal only earns money from tomorrow, so there is no peeking at the future.
8. The script scores the untouched last 40% and draws $\beta$ over time, z-scores, equity curves, and the delta sweep.

## 4. Questions an interviewer might ask

1. **Why use a Kalman filter instead of a rolling regression?** A rolling window forgets abruptly and treats all days in the window equally. The filter weights recent data smoothly and also gives a principled uncertainty estimate. In my test it gave smoother betas and smaller drawdowns on all three pairs.
2. **What does cointegration mean, and why does pairs trading need it?** Two prices can each wander like a random walk while some combination of them is stable (mean-reverting). Without that, the "gap" has no tendency to close.
3. **How did you avoid overfitting?** $R$ and $\delta$ were chosen on the first 60% only, with a minimum trade count; all reported numbers come from the last 40%. Signals use only past data, and costs are included.
4. **Your results were roughly flat to negative. Is that a failure?** It is an honest result. Only KO/PEP passed the cointegration test in training and it failed in the test window, and a 5 bps cost on a signal that trades every few days eats a small edge. The method beat the rolling baseline, but it did not find alpha on these famous pairs.
5. **What are the risks of pairs trading?** The relationship can break permanently (a merger, a product failure), so the "cheap" leg keeps getting cheaper; shorting costs money; and capacity and crowding reduce the edge.
