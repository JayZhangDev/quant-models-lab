# Marchenko-Pastur, explained

Written for someone who knows basic algebra. Code: [`marchenko_pastur.py`](../src/qml/rmt/marchenko_pastur.py), [`portfolio.py`](../src/qml/rmt/portfolio.py), [`run_rmt.py`](../experiments/run_rmt.py).

## 1. The problem in plain English

Imagine you want to know how much 300 stocks move together. You look at 1,000 days of prices and, for every pair of stocks, compute a **correlation**: a number from -1 (move opposite) through 0 (unrelated) to +1 (move together). Put them in a 300 x 300 table: the **correlation matrix**.

Here is the catch. Flip a coin 10 times twice and the two sequences will look slightly "correlated" by pure luck. With only 1,000 days, even two completely unrelated stocks show a fake correlation of a few percent. Across 44,850 pairs, those fake numbers add up to a matrix that is mostly noise.

This matters because portfolio builders trust the matrix. A "minimum-variance" optimizer hunts for the combination of stocks with the lowest risk. If the matrix contains fake patterns, the optimizer finds fake safe combinations, bets heavily on them, and gets hurt. In my run the raw matrix predicted 3.4% yearly volatility and the portfolio actually swung 19.7%.

**Real-world analogy:** a weather forecaster with only 10 days of data who "discovers" that it always rains after a Tuesday. The pattern is luck, and betting on it is a mistake. Marchenko-Pastur is a way to tell which patterns are plausibly luck.

## 2. The equations, every symbol

**Eigenvalues first.** Any correlation matrix can be split into independent "directions" (eigenvectors), each with a size (eigenvalue). The biggest direction is usually "everything moves together" (the market). A big eigenvalue means a strong pattern; the eigenvalues of a table always add up to the number of stocks N.

| Symbol | Meaning |
|---|---|
| $N$ | number of stocks (300) |
| $T$ | number of days (1,003) |
| $q = N/T$ | how data-starved you are. Must be below 1. Higher = noisier |
| $\sigma^2$ | how much of each stock's variance is noise |
| $\lambda_\pm$ | edges of the noise band |
| $\rho(\lambda)$ | how common each eigenvalue size is, if everything were noise |

$$\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{q})^2$$

$$\rho(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi q \sigma^2 \lambda}$$

**Knobs:**

- **q up** (fewer days per stock): the band $[\lambda_-, \lambda_+]$ widens, more eigenvalues look like noise. With q near 1 the lower edge hits 0.
- **q down** (lots of data): the band shrinks toward 1, so the matrix is mostly trustworthy.
- **$\sigma^2$ down**: the band slides down and narrows. Real stocks have $\sigma^2 < 1$ because one huge eigenvalue (the market) soaks up much of the fixed total N, leaving less for everything else. I fitted 0.36.

**The cleaning recipe (eigenvalue clipping):** keep eigenvalues above $\lambda_+$; replace all the others with their average; rebuild the matrix; rescale so each stock has correlation 1 with itself. Averaging keeps the total (trace) unchanged.

## 3. Walkthrough of the code, in running order

`experiments/run_rmt.py` calls things in this order:

1. `load_universe` / `get_prices` / `clean_returns` (in `common/data.py`): read the ticker list, download adjusted prices (cached), drop stocks with more than 5% missing data, turn prices into daily returns.
2. `correlation_matrix`: build the 300 x 300 table.
3. `sorted_eigh`: split it into eigenvalues and eigenvectors, biggest first.
4. `fit_sigma2`: find the $\sigma^2$ that makes the theoretical curve `mp_pdf` best match the histogram of eigenvalues. It sets aside the clear outliers first, then scans a grid and polishes with scipy.
5. `mp_bounds` and `count_signal_eigenvalues`: compute the band and count how many eigenvalues escape above it (61).
6. `clean_correlation`: clip, rebuild, renormalize.
7. Three plotting functions make the spectrum, market-eigenvector and heatmap figures.
8. `run_out_of_sample` calls `rolling_min_variance` (in `portfolio.py`): for each month, estimate on the previous 252 days (`sample_covariance` for raw, `cleaned_covariance` for cleaned), compute weights with `min_variance_weights` (the formula $w = \Sigma^{-1}\mathbf{1} / \mathbf{1}^\top\Sigma^{-1}\mathbf{1}$) or the long-only variant, then record what the portfolio did over the *next* 21 days. Nothing from the future is ever used.

## 4. Questions an interviewer might ask

1. **Why can't you just use the sample correlation matrix?** With N stocks and T days, it has about N²/2 numbers estimated from N x T data points. When N/T is not small, its eigenvalues spread far from the truth, and optimizers exploit that error.
2. **What does Marchenko-Pastur tell you?** For a pure-noise matrix, the eigenvalues fall in a known band set by q = N/T. Eigenvalues above it are statistically distinguishable from noise.
3. **Why is the fitted $\sigma^2$ below 1?** The trace of a correlation matrix is N, so a huge market eigenvalue forces the others to average below 1.
4. **What are the weaknesses of eigenvalue clipping?** It is a hard threshold that assumes the noise is uniform. It discards real small-scale structure (HD/LOW correlation fell from 0.88 to 0.70), and the estimated threshold is itself noisy. Shrinkage estimators such as Ledoit-Wolf are a common alternative.
5. **Why did cleaning barely help in the long-only case?** Banning short sales already limits how extreme the weights can get, which acts like a regularizer. In the unconstrained case, cleaning halved realized volatility (19.7% to 9.9%).
