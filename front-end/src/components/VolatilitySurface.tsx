'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
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
import { SurfaceDataPoint, VolatilitySurfaceResponse } from '@/types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ChartDataset {
  label: string;
  data: number[];
  borderColor: string;
  backgroundColor: string;
  tension: number;
  pointRadius: number;
  pointHoverRadius: number;
}

interface VolatilityChartData {
  labels: string[];
  datasets: ChartDataset[];
}

interface SurfaceParameters {
  spotPrice: number;
  riskFreeRate: number;
}

interface VolatilitySurfaceProps {
  surfaceParams?: SurfaceParameters;
  onParameterChange?: (params: SurfaceParameters) => void;
}

const VolatilitySurface: React.FC<VolatilitySurfaceProps> = ({
  surfaceParams,
  onParameterChange
}) => {
  const [parameters, setParameters] = useState<SurfaceParameters>(() => ({
    spotPrice: surfaceParams?.spotPrice || 100,
    riskFreeRate: surfaceParams?.riskFreeRate || 0.05,
  }));

  const [inputValues, setInputValues] = useState<{
    spotPrice: string;
    riskFreeRate: string;
  }>(() => ({
    spotPrice: (surfaceParams?.spotPrice || 100).toString(),
    riskFreeRate: (surfaceParams?.riskFreeRate || 0.05).toString(),
  }));

  const [surfaceData, setSurfaceData] = useState<SurfaceDataPoint[] | null>(null);
  const [chartData, setChartData] = useState<VolatilityChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (surfaceParams) {
      setParameters(surfaceParams);
      setInputValues({
        spotPrice: surfaceParams.spotPrice.toString(),
        riskFreeRate: surfaceParams.riskFreeRate.toString(),
      });
    }
  }, [surfaceParams]);

  const fetchVolatilitySurface = useCallback(async (params: SurfaceParameters): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('http://localhost:5000/api/volatility-surface', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spot_price: params.spotPrice,
          risk_free_rate: params.riskFreeRate
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: VolatilitySurfaceResponse = await response.json();

      if (!data.surface_data || !Array.isArray(data.surface_data)) {
        throw new Error('Invalid data format received');
      }

      setSurfaceData(data.surface_data);
      prepareChartData(data.surface_data);
    } catch (err) {
      console.error('Error fetching volatility surface:', err);
      setError(err instanceof Error ? err.message : 'Failed to load volatility surface data. Please check your connection and try again.');
      setSurfaceData(null);
      setChartData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (parameters.spotPrice > 0 && parameters.riskFreeRate >= 0) {
      fetchVolatilitySurface(parameters);
    }
  }, [parameters, fetchVolatilitySurface]);

  const handleParameterChange = (key: keyof SurfaceParameters, value: string): void => {
    setInputValues(prev => ({
      ...prev,
      [key]: value
    }));

    const numericValue = parseFloat(value);
    if (!isNaN(numericValue) && value.trim() !== '') {
      const newParams = {
        ...parameters,
        [key]: numericValue,
      };
      setParameters(newParams);
      onParameterChange?.(newParams);
    }
  };

  const handleInputBlur = (key: keyof SurfaceParameters): void => {
    if (inputValues[key] === '' || isNaN(parseFloat(inputValues[key]))) {
      setInputValues(prev => ({
        ...prev,
        [key]: parameters[key].toString()
      }));
    }
  };

  const prepareChartData = useCallback((data: SurfaceDataPoint[]): void => {
    try {
      const expirationGroups: Record<string, SurfaceDataPoint[]> = {};

      data.forEach(point => {
        if (point && typeof point.expiration === 'number' && typeof point.strike === 'number' && typeof point.volatility === 'number') {
          const expKey = point.expiration.toFixed(1);
          if (!expirationGroups[expKey]) {
            expirationGroups[expKey] = [];
          }
          expirationGroups[expKey].push(point);
        }
      });

      const allStrikes = [...new Set(data.map(p => p.strike))].sort((a, b) => a - b);
      const labels = allStrikes.map(strike => strike.toFixed(0));

      const colors = [
        '#ffffff', '#e5e5e5', '#cccccc', '#b3b3b3', '#999999',
        '#808080', '#666666', '#4d4d4d', '#333333'
      ];

      const datasets: ChartDataset[] = Object.keys(expirationGroups)
        .sort((a, b) => parseFloat(a) - parseFloat(b))
        .map((exp, index) => {
          const points = expirationGroups[exp].sort((a, b) => a.strike - b.strike);
          const dataArray = new Array(labels.length).fill(null);

          points.forEach(point => {
            const strikeIndex = allStrikes.indexOf(point.strike);
            if (strikeIndex !== -1) {
              dataArray[strikeIndex] = (point.volatility * 100);
            }
          });

          const color = colors[index % colors.length];

          return {
            label: `${exp}Y Expiry`,
            data: dataArray,
            borderColor: color,
            backgroundColor: `${color}20`,
            tension: 0.4,
            pointRadius: 2,
            pointHoverRadius: 4,
          };
        });

      setChartData({
        labels,
        datasets,
      });
    } catch (err) {
      console.error('Error preparing chart data:', err);
      setError('Failed to process chart data');
    }
  }, []);





    const options: ChartOptions<'line'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
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
          usePointStyle: true,
          pointStyle: 'line',
        },
      },
      title: {
        display: true,
        text: 'IMPLIED VOLATILITY SURFACE',
        color: '#ffffff',
        font: {
          family: 'Inter, system-ui, sans-serif',
          size: 16,
          weight: 600,
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#ffffff',
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y?.toFixed(2)}%`;
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Strike Price ($)',
          color: '#ffffff',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 12,
            weight: 600,
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
          lineWidth: 1,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Implied Volatility (%)',
          color: '#ffffff',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 12,
            weight: 600, 
          },
        },
        ticks: {
          color: '#cccccc',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 10,
          },
          callback: function(value) {
            return `${value}%`;
          },
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
          lineWidth: 1,
        },
        beginAtZero: false,
      },
    },
    elements: {
      line: {
        borderWidth: 2,
      },
      point: {
        radius: 2,
        hoverRadius: 4,
      },
    },
  }), []);


  

  if (loading) {
    return (
      <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
        <h2 className="text-xl font-bold mb-6 text-white tracking-wide">VOLATILITY SURFACE</h2>
        <div className="h-96 flex items-center justify-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            <div className="text-gray-400 text-sm font-mono">LOADING SURFACE...</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-black border border-red-900 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-white tracking-wide">VOLATILITY SURFACE</h2>
          <button
            onClick={() => fetchVolatilitySurface(parameters)}
            className="px-3 py-1 bg-red-900 text-white text-xs rounded hover:bg-red-800 transition-colors"
          >
            RETRY
          </button>
        </div>
        <div className="h-96 flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="text-red-400 text-sm font-mono">ERROR</div>
            <div className="text-gray-300 text-xs max-w-md">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white tracking-wide">VOLATILITY SURFACE</h2>
      </div>
      {!surfaceParams && (
        <div className="mb-4 grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">SPOT PRICE</label>
            <input
              type="number"
              value={inputValues.spotPrice}
              onChange={(e) => handleParameterChange('spotPrice', e.target.value)}
              onBlur={() => handleInputBlur('spotPrice')}
              className="w-full px-3 py-2 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
              placeholder="Current spot price"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">RISK FREE RATE</label>
            <input
              type="number"
              step="0.001"
              value={inputValues.riskFreeRate}
              onChange={(e) => handleParameterChange('riskFreeRate', e.target.value)}
              onBlur={() => handleInputBlur('riskFreeRate')}
              className="w-full px-3 py-2 bg-gray-900 text-white border border-gray-700 rounded text-sm focus:outline-none focus:ring-1 focus:ring-white"
              placeholder="Risk-free rate"
            />
          </div>
        </div>
      )}
      <div className="mb-2 text-xs text-gray-400">
        Parameters: Spot ${parameters.spotPrice} | Risk-free Rate {(parameters.riskFreeRate * 100).toFixed(2)}%
      </div>

      {chartData && chartData.datasets.length > 0 ? (
        <div className="h-96 relative bg-gray-900 rounded-lg p-4 border border-gray-800">
          <Line
            key={`surface-${parameters.spotPrice}-${parameters.riskFreeRate}`}
            data={chartData}
            options={options}
          />
        </div>
      ) : (
        <div className="h-96 flex items-center justify-center bg-gray-900 rounded-lg border border-gray-800">
          <div className="text-center">
            <div className="text-gray-400 text-sm font-mono">NO DATA AVAILABLE</div>
            <div className="text-gray-500 text-xs mt-2">Check your API endpoint</div>
          </div>
        </div>
      )}

      {surfaceData && (
        <div className="mt-4 text-xs text-gray-400 text-center">
          Surface points: {surfaceData.length} | Updated: {new Date().toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default VolatilitySurface;



