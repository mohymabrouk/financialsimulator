'use client';
import { useState, useEffect, useCallback } from 'react';

interface HedgingPosition {
  id: string;
  option_symbol: string;
  option_delta: number;
  option_gamma: number;
  option_vega: number;
  option_theta: number;
  position_size: number;
  underlying_symbol: string;
  hedge_ratio: number;
  hedge_quantity: number;
  net_delta: number;
  rebalance_threshold: number;
  current_price?: number;
  underlying_price?: number;
}

interface HedgingResults {
  total_portfolio_delta: number;
  total_portfolio_gamma: number;
  total_portfolio_vega: number;
  total_portfolio_theta: number;
  hedging_pnl: number;
  transaction_costs: number;
  hedge_effectiveness: number;
  next_rebalance_signal: string;
  last_updated: string;
}

const DeltaHedgingEngine: React.FC = () => {
  const [positions, setPositions] = useState<HedgingPosition[]>([
    {
      id: '1',
      option_symbol: 'SPY240315C00450000',
      option_delta: 0.65,
      option_gamma: 0.003,
      option_vega: 0.15,
      option_theta: -0.05,
      position_size: 100,
      underlying_symbol: 'SPY',
      hedge_ratio: -0.65,
      hedge_quantity: -65,
      net_delta: 0.05,
      rebalance_threshold: 0.1,
      current_price: 45.20,
      underlying_price: 452.50
    },
    {
      id: '2',
      option_symbol: 'QQQ240315P00380000',
      option_delta: -0.35,
      option_gamma: 0.004,
      option_vega: 0.12,
      option_theta: -0.03,
      position_size: 50,
      underlying_symbol: 'QQQ',
      hedge_ratio: 0.35,
      hedge_quantity: 18,
      net_delta: -0.02,
      rebalance_threshold: 0.08,
      current_price: 12.80,
      underlying_price: 385.75
    }
  ]);

  const [hedgingResults, setHedgingResults] = useState<HedgingResults | null>(null);
  const [autoRebalance, setAutoRebalance] = useState<boolean>(false);
  const [rebalanceFrequency, setRebalanceFrequency] = useState<number>(30);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState<string | null>(null);
  const [showAddPosition, setShowAddPosition] = useState<boolean>(false);

  const calculateHedging = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const totalDelta = positions.reduce((sum, pos) => sum + (pos.option_delta * pos.position_size) + pos.hedge_quantity, 0);
      const totalGamma = positions.reduce((sum, pos) => sum + (pos.option_gamma * pos.position_size), 0);
      const totalVega = positions.reduce((sum, pos) => sum + (pos.option_vega * pos.position_size), 0);
      const totalTheta = positions.reduce((sum, pos) => sum + (pos.option_theta * pos.position_size), 0);

      const mockResults: HedgingResults = {
        total_portfolio_delta: totalDelta,
        total_portfolio_gamma: totalGamma,
        total_portfolio_vega: totalVega,
        total_portfolio_theta: totalTheta,
        hedging_pnl: Math.random() * 1000 - 500,
        transaction_costs: positions.length * 2.5,
        hedge_effectiveness: 0.85 + Math.random() * 0.1,
        next_rebalance_signal: Math.abs(totalDelta) > 0.1 ? 'REBALANCE REQUIRED' : 'HEDGED',
        last_updated: new Date().toLocaleTimeString()
      };

      setHedgingResults(mockResults);

      setPositions(prev => prev.map(pos => ({
        ...pos,
        net_delta: (pos.option_delta * pos.position_size) + pos.hedge_quantity,
        current_price: pos.current_price ? pos.current_price + (Math.random() - 0.5) * 0.1 : undefined,
        underlying_price: pos.underlying_price ? pos.underlying_price + (Math.random() - 0.5) * 2 : undefined
      })));
    } catch (error) {
      setError('Failed to calculate hedging metrics');
      console.error('Delta hedging calculation error:', error);
    } finally {
      setIsLoading(false);
    }
  }, [positions]);

  useEffect(() => {
    if (autoRebalance && rebalanceFrequency > 0) {
      const interval = setInterval(calculateHedging, rebalanceFrequency * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRebalance, rebalanceFrequency, calculateHedging]);

  useEffect(() => {
    calculateHedging();
  }, []);

  const handleRebalancePosition = (positionId: string) => {
    setPositions(prev => prev.map(pos => {
      if (pos.id === positionId) {
        const newHedgeQty = Math.round(-(pos.option_delta * pos.position_size));
        return {
          ...pos,
          hedge_quantity: newHedgeQty,
          net_delta: 0
        };
      }
      return pos;
    }));
    calculateHedging();
  };

  const handlePositionUpdate = (positionId: string, field: keyof HedgingPosition, value: string | number) => {
    setPositions(prev => prev.map(pos => {
      if (pos.id === positionId) {
        if (typeof value === 'string' && field === 'option_symbol') {
          return { ...pos, [field]: value };
        }
        if (typeof value === 'string') {
          if (value.trim() === '') {
            return pos;
          }
          
          const numericValue = parseFloat(value);
          if (!isNaN(numericValue)) {
            return { ...pos, [field]: numericValue };}
          return pos;
        }
        return { ...pos, [field]: value };
      }
      return pos;
    }));
  };

  const addNewPosition = () => {
    const newPosition: HedgingPosition = {
      id: Date.now().toString(),
      option_symbol: 'NEW_OPTION',
      option_delta: 0.5,
      option_gamma: 0.002,
      option_vega: 0.1,
      option_theta: -0.02,
      position_size: 10,
      underlying_symbol: 'SPY',
      hedge_ratio: -0.5,
      hedge_quantity: -5,
      net_delta: 0,
      rebalance_threshold: 0.1
    };
    setPositions(prev => [...prev, newPosition]);
    setShowAddPosition(false);
  };

  const removePosition = (positionId: string) => {
    setPositions(prev => prev.filter(pos => pos.id !== positionId));
    calculateHedging();
  };

  const handleRebalanceFrequencyChange = (value: string) => {
    if (value.trim() === '') {
      return;
    }
    
    const numericValue = parseInt(value);
    if (!isNaN(numericValue) && numericValue >= 5 && numericValue <= 300) {
      setRebalanceFrequency(numericValue);
    }
  };

  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white tracking-wide">DELTA HEDGING ENGINE</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={calculateHedging}
            disabled={isLoading}
            className="px-3 py-1 bg-blue-900 text-white rounded hover:bg-blue-800 disabled:opacity-50 text-sm font-mono"
          >
            {isLoading ? 'CALCULATING...' : 'REFRESH'}
          </button>
          <button
            onClick={() => setShowAddPosition(true)}
            className="px-3 py-1 bg-green-900 text-white rounded hover:bg-green-800 text-sm font-mono"
          >
            ADD POSITION
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900 border border-red-700 rounded p-3 mb-4 text-red-200 text-sm">
          {error}
        </div>
      )}

      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-mono text-gray-300 tracking-wider">HEDGING CONTROLS</h3>
          <div className="flex items-center space-x-4">
            <label className="flex items-center text-xs font-mono text-gray-300">
              <input
                type="checkbox"
                checked={autoRebalance}
                onChange={(e) => setAutoRebalance(e.target.checked)}
                className="mr-2"
              />
              AUTO-REBALANCE
            </label>
            <input
              type="number"
              min="5"
              max="300"
              value={rebalanceFrequency}
              onChange={(e) => handleRebalanceFrequencyChange(e.target.value)}
              className="w-20 px-2 py-1 bg-black text-white border border-gray-700 rounded text-xs font-mono"
              placeholder="SEC"
            />
            <span className="text-xs text-gray-400">SEC</span>
          </div>
        </div>

        {hedgingResults && (
          <div>
            <div className="grid grid-cols-2 lg:grid-cols-8 gap-3 text-center mb-3">
              <div>
                <div className="text-xs text-gray-400 font-mono">NET DELTA</div>
                <div className={`text-sm font-bold font-mono ${
                  Math.abs(hedgingResults.total_portfolio_delta) < 0.1 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {hedgingResults.total_portfolio_delta.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">GAMMA</div>
                <div className="text-sm font-bold text-white font-mono">
                  {hedgingResults.total_portfolio_gamma.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">VEGA</div>
                <div className="text-sm font-bold text-white font-mono">
                  {hedgingResults.total_portfolio_vega.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">THETA</div>
                <div className="text-sm font-bold text-red-400 font-mono">
                  {hedgingResults.total_portfolio_theta.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">HEDGE P&L</div>
                <div className={`text-sm font-bold font-mono ${
                  hedgingResults.hedging_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  ${hedgingResults.hedging_pnl.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">TX COSTS</div>
                <div className="text-sm font-bold text-red-400 font-mono">
                  -${hedgingResults.transaction_costs.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">EFFECTIVENESS</div>
                <div className="text-sm font-bold text-white font-mono">
                  {(hedgingResults.hedge_effectiveness * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 font-mono">STATUS</div>
                <div className={`text-xs font-bold font-mono ${
                  hedgingResults.next_rebalance_signal === 'HEDGED' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {hedgingResults.next_rebalance_signal}
                </div>
              </div>
            </div>
            <div className="text-xs text-gray-500 text-center">
              Last Updated: {hedgingResults.last_updated}
            </div>
          </div>
        )}
      </div>

      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-mono text-gray-300 mb-4 tracking-wider">OPTION POSITIONS & HEDGE RATIOS</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left p-2">OPTION</th>
                <th className="text-right p-2">SIZE</th>
                <th className="text-right p-2">DELTA</th>
                <th className="text-right p-2">GAMMA</th>
                <th className="text-right p-2">PRICE</th>
                <th className="text-right p-2">HEDGE QTY</th>
                <th className="text-right p-2">NET DELTA</th>
                <th className="text-center p-2">ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={position.id} className="border-b border-gray-800 text-white hover:bg-gray-800/50">
                  <td className="p-2">
                    {isEditing === position.id ? (
                      <input
                        type="text"
                        value={position.option_symbol}
                        onChange={(e) => handlePositionUpdate(position.id, 'option_symbol', e.target.value)}
                        className="w-full px-1 py-0.5 bg-black text-white border border-gray-600 rounded text-xs"
                        onBlur={() => setIsEditing(null)}
                        onKeyDown={(e) => e.key === 'Enter' && setIsEditing(null)}
                      />
                    ) : (
                      <span
                        onClick={() => setIsEditing(position.id)}
                        className="cursor-pointer hover:text-blue-400"
                      >
                        {position.option_symbol}
                      </span>
                    )}
                  </td>
                  <td className="p-2 text-right">
                    {isEditing === position.id ? (
                      <input
                        type="number"
                        value={position.position_size}
                        onChange={(e) => handlePositionUpdate(position.id, 'position_size', e.target.value)}
                        className="w-16 px-1 py-0.5 bg-black text-white border border-gray-600 rounded text-xs text-right"
                        onBlur={() => setIsEditing(null)}
                        onKeyDown={(e) => e.key === 'Enter' && setIsEditing(null)}
                      />
                    ) : (
                      position.position_size
                    )}
                  </td>
                  <td className="p-2 text-right">{position.option_delta.toFixed(4)}</td>
                  <td className="p-2 text-right">{position.option_gamma.toFixed(4)}</td>
                  <td className="p-2 text-right text-gray-300">
                    {position.current_price?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="p-2 text-right">{position.hedge_quantity}</td>
                  <td className={`p-2 text-right ${
                    Math.abs(position.net_delta) > position.rebalance_threshold ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {position.net_delta.toFixed(4)}
                  </td>
                  <td className="p-2 text-center">
                    <div className="flex justify-center space-x-1">
                      <button
                        onClick={() => handleRebalancePosition(position.id)}
                        className="px-2 py-1 bg-blue-900 text-white rounded hover:bg-blue-800 text-xs"
                        disabled={isLoading}
                      >
                        REBAL
                      </button>
                      <button
                        onClick={() => setIsEditing(isEditing === position.id ? null : position.id)}
                        className="px-2 py-1 bg-yellow-900 text-white rounded hover:bg-yellow-800 text-xs"
                      >
                        EDIT
                      </button>
                      <button
                        onClick={() => removePosition(position.id)}
                        className="px-2 py-1 bg-red-900 text-white rounded hover:bg-red-800 text-xs"
                      >
                        DEL
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {showAddPosition && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-96">
            <h3 className="text-lg font-bold text-white mb-4">ADD NEW POSITION</h3>
            <div className="space-y-3">
              <button
                onClick={addNewPosition}
                className="w-full px-4 py-2 bg-green-900 text-white rounded hover:bg-green-800"
              >
                ADD POSITION
              </button>
              <button
                onClick={() => setShowAddPosition(false)}
                className="w-full px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
              >
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeltaHedgingEngine;
