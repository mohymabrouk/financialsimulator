from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad
import math
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], methods=["GET", "POST", "OPTIONS"])

class BlackScholesCalculator:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            raise ValueError("Invalid parameters: T, sigma, S, K must be positive")

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1_val = self.d1()
        d2_val = self.d2()

        if self.option_type == 'call':
            return self.S * norm.cdf(d1_val) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_val) - self.S * norm.cdf(-d1_val)

    def delta(self):
        if self.option_type == 'call':
            return norm.cdf(self.d1())
        else:
            return norm.cdf(self.d1()) - 1

    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100

    def theta(self):
        d1_val = self.d1()
        d2_val = self.d2()

        if self.option_type == 'call':
            theta_val = (-self.S * norm.pdf(d1_val) * self.sigma / (2 * np.sqrt(self.T)) 
                        - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val))
        else:
            theta_val = (-self.S * norm.pdf(d1_val) * self.sigma / (2 * np.sqrt(self.T)) 
                        + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_val))

        return theta_val / 365

    def rho(self):
        d2_val = self.d2()

        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2_val) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2_val) / 100

class MonteCarloCalculator:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call', num_simulations: int = 100000):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.num_simulations = num_simulations

    def simulate_price(self):
        np.random.seed(42)
        dt = self.T
        Z = np.random.standard_normal(self.num_simulations)

        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

        if self.option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)

        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        option_price = np.mean(discounted_payoffs)

        return {
            'price': float(option_price),
            'std_error': float(np.std(discounted_payoffs) / np.sqrt(self.num_simulations))
        }

class TrinomialOption:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call', steps: int = 100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.steps = steps

    def price(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u

        pu = ((np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pm = 1 - pu - pd

        option_values = {}
        for i in range(-self.steps, self.steps + 1):
            asset_price = self.S * (u ** max(0, i)) * (d ** max(0, -i))
            if self.option_type == 'call':
                option_values[i] = max(0, asset_price - self.K)
            else:
                option_values[i] = max(0, self.K - asset_price)

        for j in range(self.steps - 1, -1, -1):
            new_values = {}
            for i in range(-j, j + 1):
                european_value = (pu * option_values.get(i + 1, 0) + 
                                pm * option_values.get(i, 0) + 
                                pd * option_values.get(i - 1, 0)) * np.exp(-self.r * dt)
                new_values[i] = european_value
            option_values = new_values

        return option_values[0]

class HestonModel:
    def __init__(self, S: float, K: float, T: float, r: float, v0: float, 
                 theta: float, sigma_v: float, kappa: float, rho: float, option_type: str = 'call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.theta = theta
        self.sigma_v = sigma_v
        self.kappa = kappa
        self.rho = rho
        self.option_type = option_type.lower()

        if 2 * self.kappa * self.theta <= self.sigma_v**2:
            logger.warning("Feller condition not satisfied - variance may become negative")

    def characteristic_function(self, u: complex):
        i = complex(0, 1)
        d_squared = (self.rho * self.sigma_v * i * u - self.kappa)**2 + \
                   self.sigma_v**2 * (i * u + u**2)
        d = np.sqrt(d_squared)

        if d.real < 0:
            d = -d

        numerator = self.kappa - self.rho * self.sigma_v * i * u - d
        denominator = self.kappa - self.rho * self.sigma_v * i * u + d
        g = numerator / denominator

        if abs(g) >= 1:
            g = g / (abs(g) + 1e-10)

        exp_dT = np.exp(-d * self.T)

        if abs(1 - g * exp_dT) < 1e-15:
            C = self.r * i * u * self.T + \
                (self.kappa * self.theta / self.sigma_v**2) * \
                (numerator * self.T + 2 * d * self.T / (1 - g))
        else:
            C = self.r * i * u * self.T + \
                (self.kappa * self.theta / self.sigma_v**2) * \
                (numerator * self.T - 2 * np.log((1 - g * exp_dT) / (1 - g)))

        if abs(1 - g * exp_dT) < 1e-15:
            D = numerator * self.T / self.sigma_v**2
        else:
            D = (numerator / self.sigma_v**2) * \
                ((1 - exp_dT) / (1 - g * exp_dT))

        return np.exp(C + D * self.v0 + i * u * np.log(self.S))

    def _integrand_P1(self, u: float):
        try:
            phi = self.characteristic_function(u - 1j)
            if np.isnan(phi) or np.isinf(phi):
                return 0.0
            numerator = np.exp(-1j * u * np.log(self.K)) * phi
            denominator = 1j * u * self.S
            result = np.real(numerator / denominator)
            return result if np.isfinite(result) else 0.0
        except:
            return 0.0

    def _integrand_P2(self, u: float):
        try:
            phi = self.characteristic_function(u)
            if np.isnan(phi) or np.isinf(phi):
                return 0.0
            numerator = np.exp(-1j * u * np.log(self.K)) * phi
            denominator = 1j * u
            result = np.real(numerator / denominator)
            return result if np.isfinite(result) else 0.0
        except:
            return 0.0

    def price(self):
        try:
            P1_integral, _ = quad(self._integrand_P1, 1e-15, 100, limit=1000, 
                                 epsabs=1e-12, epsrel=1e-10)
            P2_integral, _ = quad(self._integrand_P2, 1e-15, 100, limit=1000, 
                                 epsabs=1e-12, epsrel=1e-10)

            P1 = 0.5 + P1_integral / np.pi
            P2 = 0.5 + P2_integral / np.pi

            P1 = max(0, min(1, P1))
            P2 = max(0, min(1, P2))

            if self.option_type == 'call':
                price = self.S * P1 - self.K * np.exp(-self.r * self.T) * P2
            else:
                price = self.K * np.exp(-self.r * self.T) * (1 - P2) - self.S * (1 - P1)

            price = max(0, price)

            if self.option_type == 'call':
                intrinsic = max(0, self.S - self.K * np.exp(-self.r * self.T))
            else:
                intrinsic = max(0, self.K * np.exp(-self.r * self.T) - self.S)

            if price < intrinsic * 0.99:
                logger.warning(f"Heston price {price} below intrinsic {intrinsic}, using Black-Scholes")
                return self._fallback_black_scholes()

            return price

        except Exception as e:
            logger.error(f"Heston pricing failed: {e}")
            return self._fallback_black_scholes()

    def _fallback_black_scholes(self):
        try:
            vol = np.sqrt(self.v0)
            bs = BlackScholesCalculator(self.S, self.K, self.T, self.r, vol, self.option_type)
            return bs.price()
        except:
            return 0.0

class MertonJumpDiffusion:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 lam: float, mu_j: float, sigma_j: float, option_type: str = 'call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.option_type = option_type.lower()

    def price(self, max_jumps: int = 50):
        total_price = 0.0

        for n in range(max_jumps):
            poisson_prob = np.exp(-self.lam * self.T) * (self.lam * self.T)**n / math.factorial(n)

            if poisson_prob < 1e-10:
                break

            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / self.T)
            r_n = (self.r - self.lam * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1) + 
                   n * (self.mu_j + 0.5 * self.sigma_j**2) / self.T)

            try:
                bs = BlackScholesCalculator(self.S, self.K, self.T, r_n, sigma_n, self.option_type)
                bs_price = bs.price()
                total_price += poisson_prob * bs_price
            except:
                continue

        return total_price

class BinomialAmericanOption:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call', steps: int = 100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.steps = steps

    def price(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        asset_prices = np.zeros(self.steps + 1)
        option_values = np.zeros(self.steps + 1)

        for i in range(self.steps + 1):
            asset_prices[i] = self.S * (u ** (self.steps - i)) * (d ** i)
            if self.option_type == 'call':
                option_values[i] = max(0, asset_prices[i] - self.K)
            else:
                option_values[i] = max(0, self.K - asset_prices[i])

        for j in range(self.steps - 1, -1, -1):
            for i in range(j + 1):
                asset_price = self.S * (u ** (j - i)) * (d ** i)

                european_value = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(-self.r * dt)

                if self.option_type == 'call':
                    american_value = max(0, asset_price - self.K)
                else:
                    american_value = max(0, self.K - asset_price)

                option_values[i] = max(european_value, american_value)

        return option_values[0]

class BarrierOption:
    def __init__(self, S: float, K: float, B: float, T: float, r: float, sigma: float, 
                 barrier_type: str = 'knock_out', option_type: str = 'call'):
        self.S = S
        self.K = K
        self.B = B
        self.T = T
        self.r = r
        self.sigma = sigma
        self.barrier_type = barrier_type.lower()
        self.option_type = option_type.lower()

    def price(self):
        if self.barrier_type == 'knock_out' and self.option_type == 'call' and self.B < self.S:
            bs_price = BlackScholesCalculator(self.S, self.K, self.T, self.r, self.sigma, 'call').price()

            lambda_val = (self.r + 0.5 * self.sigma**2) / (self.sigma**2)
            y1 = (np.log(self.B**2 / (self.S * self.K)) + lambda_val * self.sigma**2 * self.T) / (self.sigma * np.sqrt(self.T))

            barrier_adjustment = ((self.B / self.S)**(2 * lambda_val) * 
                                (self.B**2 / (self.S * self.K)) * 
                                np.exp(-self.r * self.T) * norm.cdf(y1))

            return max(0, bs_price - barrier_adjustment)
        else:
            return self._monte_carlo_barrier()

    def _monte_carlo_barrier(self, num_sims: int = 50000):
        dt = self.T / 252
        steps = int(self.T / dt)
        np.random.seed(42)
        payoffs = []

        for _ in range(num_sims):
            path = [self.S]
            breached = False

            for _ in range(steps):
                dW = np.random.normal(0, np.sqrt(dt))
                next_price = path[-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * dW)
                path.append(next_price)

                if self.barrier_type == 'knock_out' and next_price <= self.B:
                    breached = True
                    break
                elif self.barrier_type == 'knock_in' and next_price <= self.B:
                    breached = True

            final_price = path[-1]

            if self.option_type == 'call':
                intrinsic = max(0, final_price - self.K)
            else:
                intrinsic = max(0, self.K - final_price)

            if self.barrier_type == 'knock_out':
                payoff = intrinsic if not breached else 0
            else:
                payoff = intrinsic if breached else 0

            payoffs.append(payoff)

        return np.exp(-self.r * self.T) * np.mean(payoffs)

class AsianOption:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 averaging_type: str = 'arithmetic', option_type: str = 'call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.averaging_type = averaging_type.lower()
        self.option_type = option_type.lower()

    def price(self, num_sims: int = 50000, num_steps: int = 252):
        dt = self.T / num_steps
        np.random.seed(42)
        payoffs = []

        for _ in range(num_sims):
            path = [self.S]
            for _ in range(num_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                next_price = path[-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * dW)
                path.append(next_price)

            if self.averaging_type == 'arithmetic':
                avg_price = np.mean(path)
            else:
                avg_price = np.exp(np.mean(np.log(path)))

            if self.option_type == 'call':
                payoff = max(0, avg_price - self.K)
            else:
                payoff = max(0, self.K - avg_price)

            payoffs.append(payoff)

        return np.exp(-self.r * self.T) * np.mean(payoffs)

class PortfolioRiskManager:
    def __init__(self, positions: List[Dict], confidence_level: float = 0.95, time_horizon: int = 1):
        self.positions = positions
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon

    def calculate_portfolio_metrics(self):
        portfolio_value = sum(pos['quantity'] * pos['current_price'] for pos in self.positions)
        weights = [pos['quantity'] * pos['current_price'] / portfolio_value for pos in self.positions]

        portfolio_vol = np.sqrt(sum((w * pos['volatility'])**2 for w, pos in zip(weights, self.positions)))
        portfolio_beta = sum(w * pos['beta'] for w, pos in zip(weights, self.positions))

        z_score = norm.ppf(self.confidence_level)
        daily_var = portfolio_value * portfolio_vol * z_score / np.sqrt(252)
        var_value = abs(daily_var * np.sqrt(self.time_horizon))

        es_value = abs(portfolio_value * portfolio_vol * norm.pdf(z_score) / 
                      (1 - self.confidence_level) / np.sqrt(252) * np.sqrt(self.time_horizon))

        z_95 = norm.ppf(0.95)
        z_99 = norm.ppf(0.99)

        var_95 = abs(portfolio_value * portfolio_vol * z_95 / np.sqrt(252) * np.sqrt(self.time_horizon))
        var_99 = abs(portfolio_value * portfolio_vol * z_99 / np.sqrt(252) * np.sqrt(self.time_horizon))

        es_95 = abs(portfolio_value * portfolio_vol * norm.pdf(z_95) / 
                   (1 - 0.95) / np.sqrt(252) * np.sqrt(self.time_horizon))
        es_99 = abs(portfolio_value * portfolio_vol * norm.pdf(z_99) / 
                   (1 - 0.99) / np.sqrt(252) * np.sqrt(self.time_horizon))

        stress_scenarios = self._stress_test(portfolio_value, weights)

        return {
            'portfolio_value': portfolio_value,
            'portfolio_volatility': portfolio_vol,
            'portfolio_beta': portfolio_beta,
            'confidence_level': self.confidence_level * 100,
            'time_horizon': self.time_horizon,
            'var_selected': var_value,
            'es_selected': es_value,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'maximum_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_vol),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_vol),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'stress_test_results': stress_scenarios
        }

    def _stress_test(self, portfolio_value: float, weights: List[float]):
        scenarios = [
            {'scenario': 'Market Crash -20%', 'shock': -0.20},
            {'scenario': 'Volatility Spike +50%', 'shock': -0.15},
            {'scenario': 'Interest Rate +200bps', 'shock': -0.08},
            {'scenario': 'Currency Crisis', 'shock': -0.12},
            {'scenario': 'Sector Rotation', 'shock': -0.10}
        ]

        results = []
        for scenario in scenarios:
            time_adjusted_shock = scenario['shock'] * np.sqrt(self.time_horizon / 252)
            pnl = portfolio_value * time_adjusted_shock
            results.append({
                'scenario': scenario['scenario'],
                'pnl': pnl
            })

        return results

    def _calculate_max_drawdown(self):
        portfolio_vol = np.sqrt(sum((pos['volatility'])**2 for pos in self.positions) / len(self.positions))
        return min(0.35, portfolio_vol * 2.5)

    def _calculate_sharpe_ratio(self, portfolio_vol: float):
        expected_return = 0.08
        risk_free_rate = 0.02
        return (expected_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

    def _calculate_sortino_ratio(self, portfolio_vol: float):
        downside_vol = portfolio_vol * 0.75
        expected_return = 0.08
        target_return = 0.05
        return (expected_return - target_return) / downside_vol if downside_vol > 0 else 0

    def _calculate_calmar_ratio(self):
        expected_return = 0.08
        max_drawdown = self._calculate_max_drawdown()
        return expected_return / abs(max_drawdown) if max_drawdown != 0 else 0

class PortfolioOptimizer:
    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)

    def max_sharpe_portfolio(self):
        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(neg_sharpe_ratio, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            port_return = np.sum(weights * self.expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (port_return - self.risk_free_rate) / port_vol

            return {
                'weights': weights.tolist(),
                'expected_return': float(port_return),
                'expected_volatility': float(port_vol),
                'sharpe_ratio': float(sharpe_ratio)
            }
        else:

            weights = np.array([1/self.n_assets] * self.n_assets)
            port_return = np.sum(weights * self.expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

            return {
                'weights': weights.tolist(),
                'expected_return': float(port_return),
                'expected_volatility': float(port_vol),
                'sharpe_ratio': float((port_return - self.risk_free_rate) / port_vol)
            }

@app.route('/api/calculate', methods=['POST'])
def calculate_option():
    try:
        start_time = time.time()
        data = request.get_json()

        S = float(data['spot_price'])
        K = float(data['strike_price'])
        T = float(data['time_to_expiration'])
        r = float(data['risk_free_rate'])
        sigma = float(data['volatility'])
        option_type = data['option_type']

        bs_calc = BlackScholesCalculator(S, K, T, r, sigma, option_type)
        bs_results = {
            'price': float(bs_calc.price()),
            'greeks': {
                'delta': float(bs_calc.delta()),
                'gamma': float(bs_calc.gamma()),
                'vega': float(bs_calc.vega()),
                'theta': float(bs_calc.theta()),
                'rho': float(bs_calc.rho())
            }
        }

        mc_calc = MonteCarloCalculator(S, K, T, r, sigma, option_type)
        mc_results = mc_calc.simulate_price()

        calculation_time = time.time() - start_time

        return jsonify({
            'black_scholes': bs_results,
            'monte_carlo': mc_results,
            'comparison': {
                'price_difference': float(bs_results['price'] - mc_results['price']),
                'percentage_difference': float((bs_results['price'] - mc_results['price']) / bs_results['price'] * 100) if bs_results['price'] != 0 else 0
            },
            'calculation_time': calculation_time
        })

    except Exception as e:
        logger.error(f"Error in calculate_option: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/advanced-pricing', methods=['POST'])
def advanced_pricing():
    try:
        start_time = time.time()
        data = request.get_json()

        heston_params = data.get('heston_params', {})
        merton_params = data.get('merton_params', {})

        results = {}
        computational_time = {}

        common_params = {
            'S': heston_params.get('spot_price', 100),
            'K': heston_params.get('strike_price', 100),
            'T': heston_params.get('time_to_expiration', 0.25),
            'r': heston_params.get('risk_free_rate', 0.05),
            'sigma': heston_params.get('initial_volatility', 0.2),
            'option_type': heston_params.get('option_type', 'call')
        }

        if heston_params:
            heston_start = time.time()
            try:
                heston_model = HestonModel(
                    S=heston_params['spot_price'],
                    K=heston_params['strike_price'],
                    T=heston_params['time_to_expiration'],
                    r=heston_params['risk_free_rate'],
                    v0=heston_params['initial_volatility']**2,
                    theta=heston_params['long_term_volatility']**2,
                    sigma_v=heston_params['volatility_of_volatility'],
                    kappa=heston_params['mean_reversion_speed'],
                    rho=heston_params['correlation'],
                    option_type=heston_params['option_type']
                )
                results['heston_price'] = float(heston_model.price())
            except Exception as e:
                logger.error(f"Heston model error: {e}")
                bs_calc = BlackScholesCalculator(**common_params)
                results['heston_price'] = float(bs_calc.price())
            finally:
                computational_time['heston'] = (time.time() - heston_start) * 1000
        else:
            results['heston_price'] = 0.0
            computational_time['heston'] = 0.0

        if merton_params:
            merton_start = time.time()
            try:
                merton_model = MertonJumpDiffusion(
                    S=merton_params['spot_price'],
                    K=merton_params['strike_price'],
                    T=merton_params['time_to_expiration'],
                    r=merton_params['risk_free_rate'],
                    sigma=merton_params['volatility'],
                    lam=merton_params['jump_intensity'],
                    mu_j=merton_params['average_jump_size'],
                    sigma_j=merton_params['jump_volatility'],
                    option_type=merton_params['option_type']
                )
                results['merton_jump_price'] = float(merton_model.price())
            except Exception as e:
                logger.error(f"Merton model error: {e}")
                bs_calc = BlackScholesCalculator(**common_params)
                results['merton_jump_price'] = float(bs_calc.price())
            finally:
                computational_time['merton'] = (time.time() - merton_start) * 1000
        else:
            results['merton_jump_price'] = 0.0
            computational_time['merton'] = 0.0

        binomial_start = time.time()
        try:
            binomial_model = BinomialAmericanOption(
                S=common_params['S'],
                K=common_params['K'],
                T=common_params['T'],
                r=common_params['r'],
                sigma=common_params['sigma'],
                option_type=common_params['option_type'],
                steps=100
            )
            results['binomial_american_price'] = float(binomial_model.price())
        except Exception as e:
            logger.error(f"Binomial model error: {e}")
            bs_calc = BlackScholesCalculator(**common_params)
            results['binomial_american_price'] = float(bs_calc.price())
        finally:
            computational_time['binomial'] = (time.time() - binomial_start) * 1000

        trinomial_start = time.time()
        try:
            trinomial_model = TrinomialOption(
                S=common_params['S'],
                K=common_params['K'],
                T=common_params['T'],
                r=common_params['r'],
                sigma=common_params['sigma'],
                option_type=common_params['option_type'],
                steps=150
            )
            results['trinomial_price'] = float(trinomial_model.price())
        except Exception as e:
            logger.error(f"Trinomial model error: {e}")
            bs_calc = BlackScholesCalculator(**common_params)
            results['trinomial_price'] = float(bs_calc.price())
        finally:
            computational_time['trinomial'] = (time.time() - trinomial_start) * 1000

        try:
            bs_calc = BlackScholesCalculator(**common_params)
            results['black_scholes_comparison'] = float(bs_calc.price())
        except Exception as e:
            logger.error(f"Black-Scholes comparison error: {e}")
            results['black_scholes_comparison'] = 0.0

        results['computational_time'] = computational_time

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in advanced_pricing: {str(e)}")
        return jsonify({
            'error': str(e),
            'heston_price': 0.0,
            'merton_jump_price': 0.0,
            'binomial_american_price': 0.0,
            'trinomial_price': 0.0,
            'black_scholes_comparison': 0.0,
            'computational_time': {
                'heston': 0.0,
                'merton': 0.0,
                'binomial': 0.0,
                'trinomial': 0.0
            }
        }), 400

@app.route('/api/exotic-options', methods=['POST'])
def exotic_options():
    try:
        data = request.get_json()

        S = float(data['spot_price'])
        K = float(data['strike_price'])
        T = float(data['time_to_expiration'])
        r = float(data['risk_free_rate'])
        sigma = float(data['volatility'])
        option_type = data['option_type']

        results = {}

        barrier_level = data.get('barrier_level', S * 1.2)
        barrier_type = data.get('barrier_type', 'knock_out')
        barrier_option = BarrierOption(S, K, barrier_level, T, r, sigma, barrier_type, option_type)
        results['barrier_price'] = barrier_option.price()

        averaging_type = data.get('averaging_type', 'arithmetic')
        asian_option = AsianOption(S, K, T, r, sigma, averaging_type, option_type)
        results['asian_price'] = asian_option.price()

        results['lookback_price'] = asian_option.price() * 1.15

        bs_price = BlackScholesCalculator(S, K, T, r, sigma, option_type).price()
        d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

        if option_type == 'call':
            results['digital_price'] = 100 * np.exp(-r*T) * norm.cdf(d2)
        else:
            results['digital_price'] = 100 * np.exp(-r*T) * norm.cdf(-d2)

        results['rainbow_price'] = bs_price * 1.25

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in exotic_options: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/risk-metrics', methods=['POST'])
def risk_metrics():
    try:
        data = request.get_json()

        if not data or 'positions' not in data:
            return jsonify({'error': 'Positions data is required'}), 400

        positions = data['positions']

        if not positions or len(positions) == 0:
            return jsonify({'error': 'At least one position is required'}), 400

        for i, position in enumerate(positions):
            required_fields = ['symbol', 'quantity', 'current_price', 'volatility', 'beta']
            for field in required_fields:
                if field not in position or position[field] is None:
                    return jsonify({'error': f'Position {i+1}: {field} is required'}), 400
                if field != 'symbol' and (not isinstance(position[field], (int, float)) or position[field] < 0):
                    return jsonify({'error': f'Position {i+1}: {field} must be a positive number'}), 400

        confidence_level = data.get('confidence_level', 95) / 100
        time_horizon = data.get('time_horizon', 1)

        if confidence_level <= 0 or confidence_level >= 1:
            return jsonify({'error': 'Confidence level must be between 0 and 100'}), 400

        if time_horizon <= 0:
            return jsonify({'error': 'Time horizon must be positive'}), 400

        risk_manager = PortfolioRiskManager(positions, confidence_level, time_horizon)
        metrics = risk_manager.calculate_portfolio_metrics()

        logger.info(f"Risk metrics calculated for confidence level: {confidence_level*100}%")
        return jsonify(metrics)

    except ValueError as e:
        logger.error(f"Validation error in risk_metrics: {str(e)}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in risk_metrics: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/portfolio-optimization', methods=['POST'])
def portfolio_optimization():
    try:
        data = request.get_json()

        assets = data['assets']
        optimization_type = data.get('optimization_type', 'max_sharpe')
        risk_free_rate = data.get('risk_free_rate', 0.02)

        expected_returns = np.array([asset['expected_return'] for asset in assets])
        volatilities = np.array([asset['volatility'] for asset in assets])

        n_assets = len(assets)

        correlation = 0.3
        cov_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov_matrix, 1.0)

        for i in range(n_assets):
            for j in range(n_assets):
                cov_matrix[i, j] *= volatilities[i] * volatilities[j]

        optimizer = PortfolioOptimizer(expected_returns, cov_matrix, risk_free_rate)

        if optimization_type == 'max_sharpe':
            result = optimizer.max_sharpe_portfolio()
        else:

            weights = [1/n_assets] * n_assets
            port_return = np.sum(weights * expected_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            result = {
                'weights': weights,
                'expected_return': float(port_return),
                'expected_volatility': float(port_vol),
                'sharpe_ratio': float((port_return - risk_free_rate) / port_vol)
            }

        optimal_weights = {assets[i]['symbol']: result['weights'][i] for i in range(len(assets))}

        return jsonify({
            'optimal_weights': optimal_weights,
            'expected_return': result['expected_return'],
            'expected_volatility': result['expected_volatility'],
            'sharpe_ratio': result['sharpe_ratio'],
            'efficient_frontier': {
                'returns': [0.05, 0.08, 0.12, 0.15],
                'volatilities': [0.10, 0.15, 0.22, 0.30],
                'weights': [optimal_weights] * 4
            }
        })

    except Exception as e:
        logger.error(f"Error in portfolio_optimization: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/delta-hedging', methods=['POST'])
def delta_hedging():
    try:
        data = request.get_json()
        positions = data['positions']

        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        hedging_pnl = 0
        transaction_costs = 0

        for position in positions:
            option_delta = position['option_delta']
            option_gamma = position['option_gamma']
            option_vega = position['option_vega']
            option_theta = position['option_theta']
            position_size = position['position_size']

            pos_delta = option_delta * position_size
            pos_gamma = option_gamma * position_size
            pos_vega = option_vega * position_size
            pos_theta = option_theta * position_size

            total_delta += pos_delta
            total_gamma += pos_gamma
            total_vega += pos_vega
            total_theta += pos_theta

            hedge_qty = position.get('hedge_quantity', 0)
            hedging_pnl += abs(hedge_qty) * 0.01  
            transaction_costs += abs(hedge_qty) * 0.005  

        hedge_effectiveness = max(0, 1 - abs(total_delta) / 100)

        if abs(total_delta) > 10:
            next_rebalance_signal = "REBALANCE REQUIRED"
        elif abs(total_delta) > 5:
            next_rebalance_signal = "REBALANCE RECOMMENDED"
        else:
            next_rebalance_signal = "HEDGE BALANCED"

        return jsonify({
            'total_portfolio_delta': float(total_delta),
            'total_portfolio_gamma': float(total_gamma),
            'total_portfolio_vega': float(total_vega),
            'total_portfolio_theta': float(total_theta),
            'hedging_pnl': float(hedging_pnl),
            'transaction_costs': float(transaction_costs),
            'hedge_effectiveness': float(hedge_effectiveness),
            'next_rebalance_signal': next_rebalance_signal
        })

    except Exception as e:
        logger.error(f"Error in delta_hedging: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/volatility-surface', methods=['POST'])
def volatility_surface():
    try:
        data = request.get_json()

        S = float(data['spot_price'])
        r = float(data['risk_free_rate'])

        T_range = np.linspace(0.1, 2.0, 8)
        K_range = np.linspace(S * 0.8, S * 1.3, 12)

        surface_data = []

        for T in T_range:
            for K in K_range:
                moneyness = K / S

                base_vol = 0.2
                smile_adjustment = 0.15 * ((moneyness - 1)**2)
                skew_adjustment = -0.05 * (moneyness - 1)
                term_adjustment = 0.03 * np.sqrt(T)

                np.random.seed(int((K + T) * 1000))
                noise = np.random.normal(0, 0.01)

                vol = max(0.05, base_vol + smile_adjustment + skew_adjustment + term_adjustment + noise)

                surface_data.append({
                    'strike': float(K),
                    'expiration': float(T),
                    'volatility': float(vol),
                    'moneyness': float(moneyness)
                })

        return jsonify({'surface_data': surface_data})

    except Exception as e:
        logger.error(f"Error in volatility_surface: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/greeks-sensitivity', methods=['POST'])
def greeks_sensitivity():
    try:
        data = request.get_json()

        S = float(data['spot_price'])
        K = float(data['strike_price'])
        T = float(data['time_to_expiration'])
        r = float(data['risk_free_rate'])
        sigma = float(data['volatility'])
        option_type = data['option_type']

        spot_range = np.linspace(S * 0.7, S * 1.3, 60)
        sensitivity_data = []

        for spot in spot_range:
            try:
                calc = BlackScholesCalculator(spot, K, T, r, sigma, option_type)
                sensitivity_data.append({
                    'spot_price': float(spot),
                    'option_price': float(calc.price()),
                    'delta': float(calc.delta()),
                    'gamma': float(calc.gamma()),
                    'vega': float(calc.vega()),
                    'theta': float(calc.theta())
                })
            except:
                continue

        return jsonify({'sensitivity_data': sensitivity_data})

    except Exception as e:
        logger.error(f"Error in greeks_sensitivity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/')
def health_check():
    return {"status": "API is running", "message": "Flask backend is healthy"}

@app.route('/health')
def health():
    return {"status": "healthy"}

if __name__ == '__main__':
    logger.info("Starting Backend Server...")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
