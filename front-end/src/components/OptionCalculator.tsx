'use client';

import { useState, ChangeEvent } from 'react';
import axios, { AxiosResponse } from 'axios';
import { OptionInputs, CalculationResponse } from '@/types';

const OptionCalculator: React.FC = () => {
  const [inputs, setInputs] = useState<OptionInputs>({
    spot_price: 100,
    strike_price: 100,
    time_to_expiration: 0.25,
    risk_free_rate: 0.05,
    volatility: 0.2,
    option_type: 'call'
  });

  const [results, setResults] = useState<CalculationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>): void => {
    const { name, value } = e.target;
    setInputs(prev => ({
      ...prev,
      [name]: name === 'option_type' 
        ? value as 'call' | 'put' 
        : value === '' ? '' : parseFloat(value)
    }));
  };

  const calculateOption = async (): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      const response: AxiosResponse<CalculationResponse> = await axios.post(
        'https://financialsimulator.onrender.com/api/calculate',
        inputs
      );
      setResults(response.data);
    } catch (err) {
      console.error('Error calculating option:', err);
      setError('CALCULATION FAILED. CHECK INPUTS AND TRY AGAIN.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <h2 className="text-xl font-bold mb-6 text-white tracking-wide">OPTION PARAMETERS</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            SPOT PRICE ($)
          </label>
          <input
            type="number"
            name="spot_price"
            value={inputs.spot_price}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
            step="0.01"
            min="0"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            STRIKE PRICE ($)
          </label>
          <input
            type="number"
            name="strike_price"
            value={inputs.strike_price}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
            step="0.01"
            min="0"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            TIME TO EXPIRATION (YEARS)
          </label>
          <input
            type="number"
            name="time_to_expiration"
            value={inputs.time_to_expiration}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
            step="0.01"
            min="0.001"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            RISK-FREE RATE
          </label>
          <input
            type="number"
            name="risk_free_rate"
            value={inputs.risk_free_rate}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
            step="0.001"
            min="-1"
            max="1"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            VOLATILITY
          </label>
          <input
            type="number"
            name="volatility"
            value={inputs.volatility}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
            step="0.01"
            min="0.001"
            max="5"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-gray-300 mb-2 tracking-wider">
            OPTION TYPE
          </label>
          <select
            name="option_type"
            value={inputs.option_type}
            onChange={handleInputChange}
            className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
          >
            <option value="call">CALL</option>
            <option value="put">PUT</option>
          </select>
        </div>
      </div>

      <button
        onClick={calculateOption}
        disabled={loading}
        className="w-full bg-white text-black py-3 px-6 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-mono text-sm font-bold tracking-wider"
      >
        {loading ? 'CALCULATING...' : 'CALCULATE OPTION PRICE'}
      </button>

      {error && (
        <div className="mt-6 p-4 bg-red-900 border border-red-700 rounded-md">
          <p className="text-red-200 text-xs font-mono tracking-wider">{error}</p>
        </div>
      )}

      {results && (
        <div className="mt-8 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
              <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">BLACK-SCHOLES MODEL</h3>
              <div className="text-3xl font-bold text-white mb-6 font-mono">
                ${results.black_scholes.price.toFixed(4)}
              </div>
              <div className="space-y-3 text-xs font-mono">
                <div className="flex justify-between items-center py-2 border-b border-gray-800">
                  <span className="text-gray-400">DELTA:</span>
                  <span className="text-white font-bold">{results.black_scholes.greeks.delta.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-800">
                  <span className="text-gray-400">GAMMA:</span>
                  <span className="text-white font-bold">{results.black_scholes.greeks.gamma.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-800">
                  <span className="text-gray-400">VEGA:</span>
                  <span className="text-white font-bold">{results.black_scholes.greeks.vega.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-800">
                  <span className="text-gray-400">THETA:</span>
                  <span className="text-white font-bold">{results.black_scholes.greeks.theta.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-400">RHO:</span>
                  <span className="text-white font-bold">{results.black_scholes.greeks.rho.toFixed(4)}</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
              <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">MONTE CARLO SIMULATION</h3>
              <div className="text-3xl font-bold text-white mb-6 font-mono">
                ${results.monte_carlo.price.toFixed(4)}
              </div>
              <div className="text-xs font-mono">
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-400">STD ERROR:</span>
                  <span className="text-white font-bold">±{results.monte_carlo.std_error.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
            <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">MODEL COMPARISON</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs font-mono">
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-400">PRICE DIFFERENCE:</span>
                <span className="text-white font-bold">
                  ${Math.abs(results.comparison.price_difference).toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-400">PERCENTAGE DIFFERENCE:</span>
                <span className="text-white font-bold">
                  {Math.abs(results.comparison.percentage_difference).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptionCalculator;

