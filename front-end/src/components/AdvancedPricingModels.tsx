'use client';
import { useState, useEffect } from 'react';

interface HestonParameters {
  spot_price: number;
  strike_price: number;
  time_to_expiration: number;
  risk_free_rate: number;
  initial_volatility: number;
  long_term_volatility: number;
  volatility_of_volatility: number;
  mean_reversion_speed: number;
  correlation: number;
  option_type: 'call' | 'put';
}

interface MertonJumpParameters {
  spot_price: number;
  strike_price: number;
  time_to_expiration: number;
  risk_free_rate: number;
  volatility: number;
  jump_intensity: number;
  average_jump_size: number;
  jump_volatility: number;
  option_type: 'call' | 'put';
}

interface AdvancedPricingResults {
  heston_price: number;
  merton_jump_price: number;
  binomial_american_price: number;
  trinomial_price: number;
  black_scholes_comparison: number;
  computational_time: {
    heston: number;
    merton: number;
    binomial: number;
    trinomial: number;
  };
}

const AdvancedPricingModels: React.FC = () => {
  const [hestonParams, setHestonParams] = useState<HestonParameters>({
    spot_price: 100,
    strike_price: 100,
    time_to_expiration: 0.25,
    risk_free_rate: 0.05,
    initial_volatility: 0.2,
    long_term_volatility: 0.2,
    volatility_of_volatility: 0.3,
    mean_reversion_speed: 2.0,
    correlation: -0.5,
    option_type: 'call'
  });

  const [mertonParams, setMertonParams] = useState<MertonJumpParameters>({
    spot_price: 100,
    strike_price: 100,
    time_to_expiration: 0.25,
    risk_free_rate: 0.05,
    volatility: 0.2,
    jump_intensity: 0.1,
    average_jump_size: -0.05,
    jump_volatility: 0.15,
    option_type: 'call'
  });

  const [results, setResults] = useState<AdvancedPricingResults | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const handleNumberChange = (value: string): number | string => {
    if (value === '') return '';
    const parsed = parseFloat(value);
    return isNaN(parsed) ? '' : parsed;
  };

  const calculateAdvancedPrices = async (): Promise<void> => {
    setLoading(true);
    try {
      const processedHestonParams = {
        ...hestonParams,
        spot_price: typeof hestonParams.spot_price === 'string' ? parseFloat(hestonParams.spot_price) || 0 : hestonParams.spot_price,
        strike_price: typeof hestonParams.strike_price === 'string' ? parseFloat(hestonParams.strike_price) || 0 : hestonParams.strike_price,
        time_to_expiration: typeof hestonParams.time_to_expiration === 'string' ? parseFloat(hestonParams.time_to_expiration) || 0 : hestonParams.time_to_expiration,
        risk_free_rate: typeof hestonParams.risk_free_rate === 'string' ? parseFloat(hestonParams.risk_free_rate) || 0 : hestonParams.risk_free_rate,
        initial_volatility: typeof hestonParams.initial_volatility === 'string' ? parseFloat(hestonParams.initial_volatility) || 0 : hestonParams.initial_volatility,
        long_term_volatility: typeof hestonParams.long_term_volatility === 'string' ? parseFloat(hestonParams.long_term_volatility) || 0 : hestonParams.long_term_volatility,
        volatility_of_volatility: typeof hestonParams.volatility_of_volatility === 'string' ? parseFloat(hestonParams.volatility_of_volatility) || 0 : hestonParams.volatility_of_volatility,
        mean_reversion_speed: typeof hestonParams.mean_reversion_speed === 'string' ? parseFloat(hestonParams.mean_reversion_speed) || 0 : hestonParams.mean_reversion_speed,
        correlation: typeof hestonParams.correlation === 'string' ? parseFloat(hestonParams.correlation) || 0 : hestonParams.correlation,
      };

      const processedMertonParams = {
        ...mertonParams,
        spot_price: typeof mertonParams.spot_price === 'string' ? parseFloat(mertonParams.spot_price) || 0 : mertonParams.spot_price,
        strike_price: typeof mertonParams.strike_price === 'string' ? parseFloat(mertonParams.strike_price) || 0 : mertonParams.strike_price,
        time_to_expiration: typeof mertonParams.time_to_expiration === 'string' ? parseFloat(mertonParams.time_to_expiration) || 0 : mertonParams.time_to_expiration,
        risk_free_rate: typeof mertonParams.risk_free_rate === 'string' ? parseFloat(mertonParams.risk_free_rate) || 0 : mertonParams.risk_free_rate,
        volatility: typeof mertonParams.volatility === 'string' ? parseFloat(mertonParams.volatility) || 0 : mertonParams.volatility,
        jump_intensity: typeof mertonParams.jump_intensity === 'string' ? parseFloat(mertonParams.jump_intensity) || 0 : mertonParams.jump_intensity,
        average_jump_size: typeof mertonParams.average_jump_size === 'string' ? parseFloat(mertonParams.average_jump_size) || 0 : mertonParams.average_jump_size,
        jump_volatility: typeof mertonParams.jump_volatility === 'string' ? parseFloat(mertonParams.jump_volatility) || 0 : mertonParams.jump_volatility,
      };

      const response = await fetch('https://financialsimulator.onrender.com/api/advanced-pricing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          heston_params: processedHestonParams,
          merton_params: processedMertonParams
        })
      });

      if (!response.ok) throw new Error('Calculation failed');
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Advanced pricing error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <h2 className="text-xl font-bold mb-6 text-white tracking-wide">ADVANCED PRICING MODELS</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">HESTON STOCHASTIC VOLATILITY</h3>
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              placeholder="Initial Vol"
              value={hestonParams.initial_volatility}
              onChange={(e) => setHestonParams({
                ...hestonParams, 
                initial_volatility: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
            <input
              type="number"
              placeholder="Long-term Vol"
              value={hestonParams.long_term_volatility}
              onChange={(e) => setHestonParams({
                ...hestonParams, 
                long_term_volatility: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
            <input
              type="number"
              placeholder="Vol of Vol"
              value={hestonParams.volatility_of_volatility}
              onChange={(e) => setHestonParams({
                ...hestonParams, 
                volatility_of_volatility: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
            <input
              type="number"
              placeholder="Mean Reversion κ"
              value={hestonParams.mean_reversion_speed}
              onChange={(e) => setHestonParams({
                ...hestonParams, 
                mean_reversion_speed: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.1"
            />
            <input
              type="number"
              placeholder="Correlation ρ"
              value={hestonParams.correlation}
              min="-1"
              max="1"
              onChange={(e) => setHestonParams({
                ...hestonParams, 
                correlation: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.1"
            />
          </div>
        </div>
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">MERTON JUMP DIFFUSION</h3>
          <div className="grid grid-cols-2 gap-3">
            <input
              type="number"
              placeholder="Jump Intensity λ"
              value={mertonParams.jump_intensity}
              onChange={(e) => setMertonParams({
                ...mertonParams, 
                jump_intensity: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
            <input
              type="number"
              placeholder="Avg Jump Size μⱼ"
              value={mertonParams.average_jump_size}
              onChange={(e) => setMertonParams({
                ...mertonParams, 
                average_jump_size: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
            <input
              type="number"
              placeholder="Jump Volatility σⱼ"
              value={mertonParams.jump_volatility}
              onChange={(e) => setMertonParams({
                ...mertonParams, 
                jump_volatility: handleNumberChange(e.target.value) as number
              })}
              className="px-3 py-2 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              step="0.01"
            />
          </div>
        </div>
      </div>

      <button
        onClick={calculateAdvancedPrices}
        disabled={loading}
        className="w-full bg-white text-black py-3 px-6 rounded-md hover:bg-gray-200 disabled:opacity-50 transition-all font-mono text-sm font-bold tracking-wider mb-6"
      >
        {loading ? 'COMPUTING ADVANCED MODELS...' : 'CALCULATE ADVANCED PRICING'}
      </button>

      {results && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-gray-400 mb-2">HESTON MODEL</div>
            <div className="text-lg font-bold text-white font-mono">${results.heston_price.toFixed(4)}</div>
            <div className="text-xs text-gray-500">{results.computational_time.heston.toFixed(2)}ms</div>
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-gray-400 mb-2">MERTON JUMP</div>
            <div className="text-lg font-bold text-white font-mono">${results.merton_jump_price.toFixed(4)}</div>
            <div className="text-xs text-gray-500">{results.computational_time.merton.toFixed(2)}ms</div>
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-gray-400 mb-2">BINOMIAL AMERICAN</div>
            <div className="text-lg font-bold text-white font-mono">${results.binomial_american_price.toFixed(4)}</div>
            <div className="text-xs text-gray-500">{results.computational_time.binomial.toFixed(2)}ms</div>
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-center">
            <div className="text-xs font-mono text-gray-400 mb-2">TRINOMIAL</div>
            <div className="text-lg font-bold text-white font-mono">${results.trinomial_price.toFixed(4)}</div>
            <div className="text-xs text-gray-500">{results.computational_time.trinomial.toFixed(2)}ms</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedPricingModels;

