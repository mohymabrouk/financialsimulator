export interface OptionInputs {
  spot_price: number;
  strike_price: number;
  time_to_expiration: number;
  risk_free_rate: number;
  volatility: number;
  option_type: 'call' | 'put';
}
export interface CalculationResponse {
  black_scholes: {
    price: number;
    greeks: {
      delta: number;
      gamma: number;
      vega: number; 
      theta: number;
      rho: number;
    };
  };
  monte_carlo: {
    price: number;
    std_error: number; 
  };
  comparison: {
    price_difference: number;
    percentage_difference: number;
  };
}

export interface SensitivityDataPoint {
  spot_price: number;
  delta: number;
  gamma: number;    
  vega: number; 
  theta: number;
}

export interface SensitivityResponse {
  sensitivity_data: SensitivityDataPoint[];
}

export interface SurfaceDataPoint {
  strike: number; 
  expiration: number;
  volatility: number;
}
 
export interface VolatilitySurfaceResponse {
  surface_data: SurfaceDataPoint[];
}
export type GreekType = 'delta' | 'gamma' | 'vega' | 'theta';
export interface LineChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[]; 
    borderColor: string;
    backgroundColor: string;    
    tension: number; 
  }[];
}
