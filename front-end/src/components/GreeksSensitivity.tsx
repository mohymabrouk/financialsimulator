'use client';

import { useState, useEffect, ChangeEvent, useMemo, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { SensitivityDataPoint, SensitivityResponse, GreekType, LineChartData } from '@/types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface OptionParameters {
  spotPrice: number;
  strikePrice: number;
  timeToExpiration: number;
  riskFreeRate: number;
  volatility: number;
  optionType: 'call' | 'put';
}

interface GreeksSensitivityProps {
  optionParams?: OptionParameters;
  onParameterChange?: (params: OptionParameters) => void;
}

const GreeksSensitivity: React.FC<GreeksSensitivityProps> = ({
  optionParams,
  onParameterChange
}) => {
  const [parameters, setParameters] = useState<OptionParameters>(() => ({
    spotPrice: optionParams?.spotPrice || 100,
    strikePrice: optionParams?.strikePrice || 100,
    timeToExpiration: optionParams?.timeToExpiration || 0.25,
    riskFreeRate: optionParams?.riskFreeRate || 0.05,
    volatility: optionParams?.volatility || 0.2,
    optionType: optionParams?.optionType || 'call',
  }));
  const [inputValues, setInputValues] = useState({
    spotPrice: (optionParams?.spotPrice || 100).toString(),
    strikePrice: (optionParams?.strikePrice || 100).toString(),
    timeToExpiration: (optionParams?.timeToExpiration || 0.25).toString(),
    riskFreeRate: (optionParams?.riskFreeRate || 0.05).toString(),
    volatility: (optionParams?.volatility || 0.2).toString(),
  });

  const [sensitivityData, setSensitivityData] = useState<SensitivityDataPoint[] | null>(null);
  const [selectedGreek, setSelectedGreek] = useState<GreekType>('delta');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (optionParams) {
      setParameters(optionParams);
      setInputValues({
        spotPrice: optionParams.spotPrice.toString(),
        strikePrice: optionParams.strikePrice.toString(),
        timeToExpiration: optionParams.timeToExpiration.toString(),
        riskFreeRate: optionParams.riskFreeRate.toString(),
        volatility: optionParams.volatility.toString(),
      });
    }
  }, [optionParams]);

  const fetchSensitivityData = useCallback(async (params: OptionParameters): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('http://localhost:5000/api/greeks-sensitivity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spot_price: params.spotPrice,
          strike_price: params.strikePrice,
          time_to_expiration: params.timeToExpiration,
          risk_free_rate: params.riskFreeRate,
          volatility: params.volatility,
          option_type: params.optionType,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SensitivityResponse = await response.json();

      if (!data.sensitivity_data || data.sensitivity_data.length === 0) {
        throw new Error('No sensitivity data received');
      }

      setSensitivityData(data.sensitivity_data);
    } catch (err) {
      console.error('Error fetching sensitivity data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load sensitivity data');
      setSensitivityData(null);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    if (
      !isNaN(parameters.spotPrice) && parameters.spotPrice > 0 &&
      !isNaN(parameters.strikePrice) && parameters.strikePrice > 0 &&
      !isNaN(parameters.timeToExpiration) && parameters.timeToExpiration > 0 &&
      !isNaN(parameters.riskFreeRate) &&
      !isNaN(parameters.volatility) && parameters.volatility > 0
    ) {
      fetchSensitivityData(parameters);
    }
  }, [parameters, fetchSensitivityData]);

  const handleGreekChange = (e: ChangeEvent<HTMLSelectElement>): void => {
    setSelectedGreek(e.target.value as GreekType);
  };

  const handleParameterChange = (key: keyof OptionParameters, value: string): void => {
    if (key === 'optionType') {
      setParameters(prev => ({ ...prev, [key]: value as 'call' | 'put' }));
      onParameterChange?.({ ...parameters, [key]: value as 'call' | 'put' });
      return;
    }
    setInputValues(prev => ({ ...prev, [key]: value }));
    const numericValue = parseFloat(value);
    if (!isNaN(numericValue)) {
      const newParams = { ...parameters, [key]: numericValue };
      setParameters(newParams);
      onParameterChange?.(newParams);
    }
  };

  const handleInputBlur = (key: keyof OptionParameters) => {
    const currentInputValue = inputValues[key as keyof typeof inputValues];
    const numericValue = parseFloat(currentInputValue);
    
    if (isNaN(numericValue) || currentInputValue === '') {
      setInputValues(prev => ({
        ...prev,
        [key]: parameters[key].toString()
      }));
    }
  };

  const chartData = useMemo((): LineChartData | null => {
    if (!sensitivityData || sensitivityData.length === 0) return null;

    return {
      labels: sensitivityData.map(d => d.spot_price.toFixed(1)),
      datasets: [
        {
          label: selectedGreek.charAt(0).toUpperCase() + selectedGreek.slice(1),
          data: sensitivityData.map(d => d[selectedGreek]),
          borderColor: '#ffffff',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 1,
          pointHoverRadius: 4,
          pointBackgroundColor: '#ffffff',
          pointBorderColor: '#ffffff',
        } as any,
      ],
    };
  }, [sensitivityData, selectedGreek]);
  
  const chartOptions = useMemo((): ChartOptions<'line'> => ({
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    intersect: false,
    mode: 'index',
  },
  plugins: {
    legend: {
      position: 'top' as const,
      labels: {
        color: '#ffffff',
        font: {
          family: 'Inter, system-ui, sans-serif',
          size: 12,
        },
      },
    },
    title: {
      display: true,
      text: `${selectedGreek.charAt(0).toUpperCase() + selectedGreek.slice(1)} Sensitivity Analysis`,
      color: '#ffffff',
      font: {
        family: 'Inter, system-ui, sans-serif',
        size: 16,
        weight: 600,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#ffffff',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Spot Price ($)',
          color: '#ffffff',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 12,
          },
        },
        ticks: {
          color: '#cccccc',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 10,
          },
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
      y: {
        title: {
          display: true,
          text: selectedGreek.charAt(0).toUpperCase() + selectedGreek.slice(1),
          color: '#ffffff',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 12,
          },
        },
        ticks: {
          color: '#cccccc',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 10,
          },
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
    },
  }), [selectedGreek]);

  if (loading) {
    return (
      <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-white tracking-wide">GREEKS SENSITIVITY</h2>
        </div>
        <div className="h-80 flex items-center justify-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            <div className="text-gray-400 text-sm font-mono">LOADING DATA...</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-black border border-red-900 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-white tracking-wide">GREEKS SENSITIVITY</h2>
          <button
            onClick={() => fetchSensitivityData(parameters)}
            className="px-3 py-1 bg-red-900 text-white text-xs rounded hover:bg-red-800 transition-colors"
          >
            RETRY
          </button>
        </div>
        <div className="h-80 flex items-center justify-center">
          <div className="text-center">
            <div className="text-red-400 text-sm font-mono mb-2">ERROR</div>
            <div className="text-gray-300 text-xs max-w-md">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white tracking-wide">GREEKS SENSITIVITY</h2>
        <div className="flex items-center space-x-4">
          <select
            value={selectedGreek}
            onChange={handleGreekChange}
            className="px-4 py-2 bg-gray-900 text-white border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent font-mono text-sm"
          >
            <option value="delta">DELTA</option>
            <option value="gamma">GAMMA</option>
            <option value="vega">VEGA</option>
            <option value="theta">THETA</option>
          </select>
        </div>
      </div>

      {!optionParams && (
        <div className="mb-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">SPOT</label>
            <input
              type="number"
              value={inputValues.spotPrice}
              onChange={(e) => handleParameterChange('spotPrice', e.target.value)}
              onBlur={() => handleInputBlur('spotPrice')}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">STRIKE</label>
            <input
              type="number"
              value={inputValues.strikePrice}
              onChange={(e) => handleParameterChange('strikePrice', e.target.value)}
              onBlur={() => handleInputBlur('strikePrice')}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">TIME</label>
            <input
              type="number"
              step="0.01"
              value={inputValues.timeToExpiration}
              onChange={(e) => handleParameterChange('timeToExpiration', e.target.value)}
              onBlur={() => handleInputBlur('timeToExpiration')}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">RATE</label>
            <input
              type="number"
              step="0.001"
              value={inputValues.riskFreeRate}
              onChange={(e) => handleParameterChange('riskFreeRate', e.target.value)}
              onBlur={() => handleInputBlur('riskFreeRate')}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">VOL</label>
            <input
              type="number"
              step="0.01"
              value={inputValues.volatility}
              onChange={(e) => handleParameterChange('volatility', e.target.value)}
              onBlur={() => handleInputBlur('volatility')}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">TYPE</label>
            <select
              value={parameters.optionType}
              onChange={(e) => handleParameterChange('optionType', e.target.value)}
              className="w-full px-2 py-1 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
            >
              <option value="call">CALL</option>
              <option value="put">PUT</option>
            </select>
          </div>
        </div>
      )}

      <div className="mb-2 text-xs text-gray-400">
        Current Spot: ${parameters.spotPrice} | Range: ${(parameters.spotPrice * 0.7).toFixed(1)} - ${(parameters.spotPrice * 1.3).toFixed(1)}
      </div>

      {chartData && (
        <div className="h-80 relative bg-gray-900 rounded-lg p-4 border border-gray-800">
          <Line
            key={`${selectedGreek}-${parameters.spotPrice}-${parameters.strikePrice}-${parameters.timeToExpiration}-${parameters.volatility}-${parameters.optionType}`}
            data={chartData}
            options={chartOptions}
          />
        </div>
      )}

      {sensitivityData && (
        <div className="mt-4 text-xs text-gray-400 text-center">
          Data points: {sensitivityData.length} | Updated: {new Date().toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default GreeksSensitivity;


