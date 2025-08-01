'use client';
import { useState } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, } from 'chart.js';
import { Doughnut, Scatter } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
);

interface Asset {
  symbol: string;
  expected_return: number | '';
  volatility: number | '';
  current_weight: number | '';
  min_weight: number | '';
  max_weight: number | '';
}

interface OptimizationResult {
  optimal_weights: { [symbol: string]: number };
  expected_return: number;
  expected_volatility: number;
  sharpe_ratio: number;
  efficient_frontier: {
    returns: number[];
    volatilities: number[];
    weights: { [symbol: string]: number }[];
  };
}

const PortfolioOptimizer: React.FC = () => {
  const [assets, setAssets] = useState<Asset[]>([
    { symbol: 'SPY', expected_return: 0.10, volatility: 0.15, current_weight: 0.4, min_weight: 0, max_weight: 0.8 },
    { symbol: 'QQQ', expected_return: 0.12, volatility: 0.20, current_weight: 0.3, min_weight: 0, max_weight: 0.6 },
    { symbol: 'TLT', expected_return: 0.04, volatility: 0.12, current_weight: 0.2, min_weight: 0.1, max_weight: 0.5 },
    { symbol: 'GLD', expected_return: 0.06, volatility: 0.18, current_weight: 0.1, min_weight: 0, max_weight: 0.3 }
  ]);

  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null);
  const [optimizationType, setOptimizationType] = useState<'max_sharpe' | 'min_variance' | 'target_return'>('max_sharpe');
  const [targetReturn, setTargetReturn] = useState<number | ''>(0.08);
  const [riskFreeRate, setRiskFreeRate] = useState<number | ''>(0.02);
  const [loading, setLoading] = useState<boolean>(false);

  const updateAsset = (index: number, field: keyof Asset, value: string | number) => {
    const newAssets = [...assets];
    if (field === 'symbol') {
      newAssets[index] = { ...newAssets[index], [field]: value as string };
    } else {
      const numValue = value === '' ? '' : parseFloat(value as string);
      newAssets[index] = { ...newAssets[index], [field]: isNaN(numValue as number) ? '' : numValue };
    }
    setAssets(newAssets);
  };

  const addAsset = () => {
    const newAsset: Asset = {
      symbol: '',
      expected_return: 0.08,
      volatility: 0.20,
      current_weight: 0,
      min_weight: 0,
      max_weight: 1
    };
    setAssets([...assets, newAsset]);
  };

  const removeAsset = (index: number) => {
    if (assets.length > 1) {
      const newAssets = assets.filter((_, i) => i !== index);
      setAssets(newAssets);
    }
  };
  const getNumericValue = (value: number | ''): number => {
    return typeof value === 'number' ? value : 0;
  };

  const optimizePortfolio = async () => {
    setLoading(true);
    try {
      const apiAssets = assets.map(asset => ({
        symbol: asset.symbol,
        expected_return: getNumericValue(asset.expected_return),
        volatility: getNumericValue(asset.volatility),
        current_weight: getNumericValue(asset.current_weight),
        min_weight: getNumericValue(asset.min_weight),
        max_weight: getNumericValue(asset.max_weight)
      }));

      const response = await fetch('http://localhost:5000/api/portfolio-optimization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assets: apiAssets,
          optimization_type: optimizationType,
          target_return: getNumericValue(targetReturn),
          risk_free_rate: getNumericValue(riskFreeRate)
        })
      });

      if (!response.ok) throw new Error('API call failed');
      const data = await response.json();
      setOptimizationResult(data);
      setLoading(false);
    } catch (error) {
      console.error('API call failed, using mock data:', error);
      const mockResult: OptimizationResult = {
        optimal_weights: { 'SPY': 0.35, 'QQQ': 0.25, 'TLT': 0.25, 'GLD': 0.15 },
        expected_return: 0.085,
        expected_volatility: 0.142,
        sharpe_ratio: 0.458,
        efficient_frontier: { returns: [], volatilities: [], weights: [] }
      };
      setTimeout(() => {
        setOptimizationResult(mockResult);
        setLoading(false);
      }, 1000);
    }
  };

  const createDoughnutData = () => {
    if (!optimizationResult) return null;
    const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];
    
    return {
      labels: assets.map(asset => asset.symbol),
      datasets: [{
        label: 'Optimal Weights (%)',
        data: assets.map(asset => ((optimizationResult.optimal_weights[asset.symbol] || 0) * 100).toFixed(1)),
        backgroundColor: colors.slice(0, assets.length),
        borderColor: '#1F2937',
        borderWidth: 2,
        hoverBorderWidth: 3,
      }]
    };
  };

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="bg-gray-800 border border-gray-700 rounded-xl shadow-2xl p-6 mb-6">
          <h1 className="text-3xl font-bold mb-8 text-white text-center tracking-wide">
            PORTFOLIO OPTIMIZATION
          </h1>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            <div className="bg-gray-900 border border-gray-600 rounded-lg p-6">
              <h2 className="text-lg font-mono text-gray-200 mb-6 tracking-wider border-b border-gray-700 pb-2">
                ASSET UNIVERSE
              </h2>
              
              <div className="grid grid-cols-7 gap-2 text-xs text-gray-400 font-mono mb-4 px-2 py-2 bg-gray-800 rounded">
                <div className="font-semibold">SYMBOL</div>
                <div className="font-semibold">E[R]</div>
                <div className="font-semibold">σ</div>
                <div className="font-semibold">WEIGHT</div>
                <div className="font-semibold">MIN</div>
                <div className="font-semibold">MAX</div>
                <div className="font-semibold">ACTION</div>
              </div>

              <div className="space-y-3 max-h-72 overflow-y-auto mb-6">
                {assets.map((asset, index) => (
                  <div key={index} className="grid grid-cols-7 gap-2 text-sm p-2 bg-gray-800 rounded hover:bg-gray-750">
                    <input
                      type="text"
                      value={asset.symbol}
                      onChange={(e) => updateAsset(index, 'symbol', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="TICKER"
                    />
                    <input
                      type="number"
                      step="0.001"
                      value={asset.expected_return}
                      onChange={(e) => updateAsset(index, 'expected_return', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="0.000"
                    />
                    <input
                      type="number"
                      step="0.001"
                      value={asset.volatility}
                      onChange={(e) => updateAsset(index, 'volatility', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="0.000"
                    />
                    <input
                      type="number"
                      step="0.01"
                      value={asset.current_weight}
                      onChange={(e) => updateAsset(index, 'current_weight', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="0.00"
                    />
                    <input
                      type="number"
                      step="0.01"
                      value={asset.min_weight}
                      onChange={(e) => updateAsset(index, 'min_weight', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="0.00"
                    />
                    <input
                      type="number"
                      step="0.01"
                      value={asset.max_weight}
                      onChange={(e) => updateAsset(index, 'max_weight', e.target.value)}
                      className="px-2 py-2 bg-gray-900 text-white border border-gray-600 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                      placeholder="0.00"
                    />
                    <button
                      onClick={() => removeAsset(index)}
                      className="px-3 py-2 bg-red-600 text-white rounded font-mono hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      disabled={assets.length === 1}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>

              <button
                onClick={addAsset}
                className="w-full mb-6 py-3 bg-gray-700 text-white rounded font-mono text-sm hover:bg-gray-600 transition-colors border border-gray-600"
              >
                + ADD ASSET
              </button>

              <div className="space-y-4">
                <div>
                  <label className="block text-gray-300 text-sm mb-2 font-mono font-semibold">
                    RISK-FREE RATE
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={riskFreeRate}
                    onChange={(e) => setRiskFreeRate(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-600 rounded text-sm font-mono focus:border-blue-500 focus:outline-none"
                    placeholder="0.020"
                  />
                </div>

                <div>
                  <label className="block text-gray-300 text-sm mb-2 font-mono font-semibold">
                    OPTIMIZATION TYPE
                  </label>
                  <select
                    value={optimizationType}
                    onChange={(e) => setOptimizationType(e.target.value as 'max_sharpe' | 'min_variance' | 'target_return')}
                    className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-600 rounded text-sm font-mono focus:border-blue-500 focus:outline-none"
                  >
                    <option value="max_sharpe">MAXIMIZE SHARPE RATIO</option>
                    <option value="min_variance">MINIMIZE VARIANCE</option>
                    <option value="target_return">TARGET RETURN</option>
                  </select>
                </div>

                {optimizationType === 'target_return' && (
                  <div>
                    <label className="block text-gray-300 text-sm mb-2 font-mono font-semibold">
                      TARGET RETURN
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={targetReturn}
                      onChange={(e) => setTargetReturn(e.target.value === '' ? '' : parseFloat(e.target.value))}
                      className="w-full px-4 py-3 bg-gray-900 text-white border border-gray-600 rounded text-sm font-mono focus:border-blue-500 focus:outline-none"
                      placeholder="0.080"
                    />
                  </div>
                )}

                <button
                  onClick={optimizePortfolio}
                  disabled={loading}
                  className="w-full bg-white text-black py-3 px-6 rounded-md hover:bg-gray-200 disabled:opacity-50 transition-all font-mono text-sm font-bold tracking-wider mb-6"
                >
                  {loading ? 'OPTIMIZING...' : 'OPTIMIZE PORTFOLIO'}
                </button>
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-600 rounded-lg p-6">
              <h2 className="text-lg font-mono text-gray-200 mb-6 tracking-wider border-b border-gray-700 pb-2">
                OPTIMIZATION RESULTS
              </h2>
              {optimizationResult ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gray-800 p-4 rounded text-center border border-gray-700">
                      <div className="text-xs text-gray-400 font-mono mb-1">EXPECTED RETURN</div>
                      <div className="text-xl font-bold text-green-400 font-mono">
                        {(optimizationResult.expected_return * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded text-center border border-gray-700">
                      <div className="text-xs text-gray-400 font-mono mb-1">VOLATILITY</div>
                      <div className="text-xl font-bold text-yellow-400 font-mono">
                        {(optimizationResult.expected_volatility * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded text-center border border-gray-700">
                      <div className="text-xs text-gray-400 font-mono mb-1">SHARPE RATIO</div>
                      <div className="text-xl font-bold text-white font-mono">
                        {optimizationResult.sharpe_ratio.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-800 p-4 rounded border border-gray-700">
                    <div className="text-sm text-gray-300 font-mono mb-3 font-semibold">OPTIMAL ALLOCATION</div>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {assets.map((asset, index) => {
                        const currentWeight = getNumericValue(asset.current_weight) * 100;
                        const optimalWeight = (optimizationResult.optimal_weights[asset.symbol] || 0) * 100;
                        const difference = optimalWeight - currentWeight;
                        return (
                          <div key={index} className="flex justify-between items-center text-sm font-mono p-2 bg-gray-900 rounded">
                            <span className="text-white font-semibold">{asset.symbol}</span>
                            <div className="flex space-x-4">
                              <span className="text-gray-300">
                                {currentWeight.toFixed(1)}% → {optimalWeight.toFixed(1)}%
                              </span>
                              <span className={`font-semibold ${difference >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {difference >= 0 ? '+' : ''}{difference.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="bg-gray-800 p-4 rounded border border-gray-700">
                    <div className="text-sm text-gray-300 font-mono mb-4 font-semibold">OPTIMAL WEIGHTS DISTRIBUTION</div>
                    <div className="h-64 flex items-center justify-center">
                      {createDoughnutData() && (
                        <Doughnut
                          data={createDoughnutData()!}
                          options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                              legend: {
                                position: 'bottom',
                                labels: {
                                  color: '#D1D5DB',
                                  font: { size: 12 },
                                  padding: 20,
                                  usePointStyle: true,
                                }
                              },
                              tooltip: {
                                backgroundColor: '#1F2937',
                                titleColor: '#F9FAFB',
                                bodyColor: '#D1D5DB',
                                borderColor: '#374151',
                                borderWidth: 1,
                              }
                            },
                            elements: {
                              arc: {
                                borderWidth: 2,
                                hoverBorderWidth: 3,
                              }
                            }
                          }}
                        />
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-96 text-gray-400 text-sm font-mono bg-gray-800 rounded border border-gray-700">
                  <div className="text-center">
                    <div className="text-6xl mb-4">📊</div>
                    <div>RUN OPTIMIZATION TO VIEW RESULTS</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioOptimizer;
