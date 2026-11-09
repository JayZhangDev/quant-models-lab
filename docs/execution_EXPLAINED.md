# Almgren-Chriss optimal execution, explained

Code: [`almgren_chriss.py`](../src/qml/execution/almgren_chriss.py), [`frontier.py`](../src/qml/execution/frontier.py), [`run_almgren_chriss.py`](../experiments/run_almgren_chriss.py).

## 1. The problem in plain English

A pension fund wants to sell 1,000,000 shares of a stock. Dump them all in one minute and the price tanks as buyers run out; you get far less than the quoted price. Sell one share a day for years and the price may wander away before you finish.

So there are two enemies:

- **Market impact:** trading fast pushes the price against you.
- **Timing risk:** trading slowly leaves you holding shares while the price moves randomly.

Almgren and Chriss (2000) ask: given how much you dislike risk, what is the best schedule of how many shares to sell in each time slice?

## 2. The equations, every symbol

| Symbol | Meaning |
|---|---|
| $X$ | shares to sell (1,000,000) |
| $T$ | time allowed (5 days) |
| $x(t)$ | shares still held at time $t$ |
| $\sigma$ | price volatility (dollars per share per sqrt(day)) |
| $\eta$ | temporary impact: extra cost per share per unit of trading speed |
| $\gamma$ | permanent impact: how far the price moves for good per share sold |
| $\lambda$ | your risk aversion |
| $C$ | total cost: shares' value at the starting price minus cash received |

$$\min_x \; E[C] + \lambda\,\mathrm{Var}[C]$$

$$\kappa = \sqrt{\frac{\lambda\sigma^2}{\eta}}, \qquad x(t) = X\,\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}$$

**Knobs:**

- **$\lambda = 0$:** you do not care about risk, only average cost. $\kappa = 0$ and the schedule is a straight line, selling the same amount every interval (**TWAP**, time-weighted average price).
- **$\lambda$ up:** $\kappa$ grows and the curve bends: sell more at the start, then taper. A very nervous trader nearly finishes within the first day.
- **$\sigma$ up:** same as raising $\lambda$. Price swings make waiting scarier.
- **$\eta$ up:** impact is expensive, so you trade more slowly (smaller $\kappa$).
- **$\gamma$:** permanent impact costs $\gamma X^2/2$ whatever schedule you pick, so it does not change the shape (only through a small discrete-time correction).

In plain terms, $1/\kappa$ is the natural time scale of the sale.

The code uses the exact *discrete* version for 65 half-hour slices. Its derivation is in the module docstring: set the derivative of the cost to zero, which gives a recurrence, then solve it with $\cosh$. Tests confirm that as the slices get finer it matches the continuous formula above.

**Efficient frontier:** every $\lambda$ gives a schedule with an expected cost and a risk. Plotting all of them gives a curve; no schedule can beat it.

**Square-root impact:** real markets are often better described by impact proportional to $\sqrt{\text{speed}}$. There is no closed-form solution, so the code finds the answer with a numerical optimizer under the constraints "sell exactly $X$" and "never buy".

## 3. Walkthrough of the code, in running order

1. `ACParams` holds the example: 1M shares, 5 days, $50 price, $\sigma = 0.95$ per day, $\eta = 2.5\times10^{-6}$, $\gamma = 2.5\times10^{-7}$, half-spread $0.0625$.
2. `kappa_discrete` computes the discrete urgency from the $\cosh$ equation; `optimal_trajectory` evaluates $x_j = X \sinh(\kappa(T - t_j))/\sinh(\kappa T)$ (written with exponentials to avoid overflow).
3. `expected_cost_and_variance` gives the closed-form mean (permanent + half-spread + temporary impact) and variance ($\sigma^2\tau\sum x_k^2$, driven by the shares still held).
4. `frontier.py`: `frontier_point` and `efficient_frontier` loop over a log grid of $\lambda$ and collect the cost and variance for each schedule.
5. `optimize_trajectory` solves the non-closed-form problems with scipy's SLSQP. It works with fractions of $X$ and rescales the objective so numbers stay near 1, and supplies exact gradients. With linear impact it must reproduce the closed form, and a test checks that. `sqrt_impact_coefficient` scales the square-root model to cost the same as the linear model at TWAP speed, so the comparison is about shape, not price level.
6. `simulate_costs` draws 10,000 random price paths, executes a schedule on each (price shocks plus permanent impact, and execution below the last price by half-spread plus temporary impact) and records the shortfall. The experiment checks that the simulated mean and std match the formulas (within 0.4 standard errors).

## 4. Questions an interviewer might ask

1. **What is the tradeoff in Almgren-Chriss?** Fast selling pays more market impact but has less price risk; slow selling is cheaper in impact but riskier. $\lambda$ sets the exchange rate.
2. **What is temporary versus permanent impact?** Temporary impact only affects the price you get on that trade and fades; permanent impact shifts the market price for everyone afterwards. With linear permanent impact, total permanent cost does not depend on the schedule.
3. **What is TWAP and when is it optimal?** Selling at a constant rate. It is optimal under linear impact when the trader is risk-neutral ($\lambda = 0$).
4. **What changes with square-root impact?** The cost of trading fast grows more slowly, so the optimizer sells more up front. At the same $\lambda$ it opened with 100,000 shares versus 45,000 and had sold 68% after one day versus 46%.
5. **What are the limitations?** Constant volatility and liquidity (real volume peaks at the open and close), a fixed schedule that never reacts to prices, a mean-variance definition of risk, and impact parameters that must be estimated. My square-root scaling is a convenience, not a calibration.
