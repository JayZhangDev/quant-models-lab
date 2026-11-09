# Avellaneda-Stoikov market making, explained

Code: [`simulator.py`](../src/qml/market_making/simulator.py), [`strategies.py`](../src/qml/market_making/strategies.py), [`run_avellaneda_stoikov.py`](../experiments/run_avellaneda_stoikov.py).

## 1. The problem in plain English

A **market maker** is like a currency-exchange booth at an airport. They post two prices: a **bid** (what they will pay to buy) and an **ask** (what they will charge to sell). The ask is higher than the bid. Each time someone buys at the ask and someone else sells at the bid, the booth keeps the gap (the **spread**).

The risk: if many customers sell you euros and nobody buys, you pile up euros (**inventory**). If the euro then drops, you lose money. So a good booth owner who is stuck with too many euros lowers *both* prices a bit: the cheaper ask attracts buyers, the cheaper bid discourages more sellers. Avellaneda and Stoikov (2008) turned that instinct into exact formulas.

## 2. The equations, every symbol

| Symbol | Meaning |
|---|---|
| $s$ | current mid-price (halfway between bid and ask) |
| $q$ | inventory: shares held (negative means you are short) |
| $\gamma$ | risk aversion. High = you hate holding inventory |
| $\sigma$ | how violently the price moves |
| $T - t$ | time left in the session |
| $k$ | how fast customers lose interest as your quote gets worse |
| $A$ | how many customers arrive if you quote exactly at the mid |

**Reservation price** (the price at which you are neutral about trading):

$$r = s - q\,\gamma\,\sigma^2\,(T - t)$$

**Total spread:**

$$\delta^a + \delta^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$$

Quotes: bid = $r$ - spread/2, ask = $r$ + spread/2.

**Fill rate:** $\lambda(\delta) = A e^{-k\delta}$. A quote $\delta$ away from the mid gets fewer customers the further away it is.

**Knobs:**

- **Inventory $q$ up:** $r$ drops further below $s$. Both quotes slide down. You lean toward selling.
- **$\gamma$ up:** bigger lean and a wider spread. Less inventory risk, but fewer trades, so lower profit.
- **$\sigma$ up:** same effect as raising $\gamma$. A wilder price means more to fear from holding stock.
- **Time left $T - t$ down:** the lean fades and the first spread term shrinks to zero at the close. With no time left, inventory cannot lose more.
- **$k$ up:** customers are very price-sensitive, so you must quote tight.
- **$\sigma = 0$:** no price risk, so $r = s$ whatever your inventory. A test checks this.

## 3. Walkthrough of the code, in running order

1. `ASParams` (in `simulator.py`) holds the paper's defaults: $s_0 = 100$, $T = 1$, $\sigma = 2$, $dt = 0.005$, $\gamma = 0.1$, $k = 1.5$, $A = 140$.
2. `simulate(strategy, params, n_paths, seed)` first draws all random numbers up front: price up/down moves and two uniform draws per step for "did a bid fill?" and "did an ask fill?". Drawing them once means both strategies face identical luck, so their comparison is fair.
3. A loop over the 200 time steps (vectorized across all 1,000 paths) does:
   - ask the strategy for quotes (`strategy.quotes`);
   - compute fill probability = `fill_intensity` x `dt` for each side;
   - compare against the stored uniform draw to decide who got filled;
   - update inventory and cash (cash falls by the bid when we buy, rises by the ask when we sell);
   - move the mid-price one step.
4. The final profit is cash plus inventory valued at the last mid-price (`SimulationResult.pnl`).
5. `strategies.py`: `reservation_price` and `optimal_spread` implement the formulas. `InventoryStrategy` centers on $r$; `SymmetricStrategy` uses the same spread but centers on $s$. That is the only difference, which is what makes the comparison clean.
6. `run_avellaneda_stoikov.py` runs both for $\gamma \in \{0.01, 0.1, 0.5\}$, builds the table, and plots one path, profit histograms and final-inventory histograms.

## 4. Questions an interviewer might ask

1. **What is the reservation price and why does it depend on inventory?** It is the price at which the market maker's expected utility is the same whether or not they trade one more share. A long position means holding unwanted risk, so the maker is willing to sell for less than the mid-price.
2. **What happens to the spread as time runs out?** The risk term $\gamma\sigma^2(T-t)$ falls to zero, so the spread shrinks toward its constant part, which reflects only the fill-rate tradeoff.
3. **What did the simulation show?** Inventory-aware quotes cut the standard deviation of profit and of final inventory roughly in half at $\gamma = 0.1$, for about 5% lower mean profit.
4. **What assumptions are unrealistic?** The mid-price is a driftless random walk, order arrivals follow a smooth exponential curve, quotes fill one share at a time, and there is no adverse selection (informed traders), queue position, fees, or tick size.
5. **How would you choose $\gamma$ in practice?** It is a business decision: how much variance of profit you will give up expected profit to avoid. The table shows the price tag: 1% of mean profit at $\gamma = 0.01$ and 18% at $\gamma = 0.5$.
