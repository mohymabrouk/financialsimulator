'use client';

import dynamic from 'next/dynamic';
import { Suspense } from 'react';
const OptionCalculator = dynamic(() => import('@/components/OptionCalculator'), {
  loading: () => <ComponentLoader />
});

const DeltaHedgingEngine = dynamic(() => import('@/components/DeltaHedgingEngine'), {
  loading: () => <ComponentLoader />
});

const AdvancedPricingModels = dynamic(() => import('@/components/AdvancedPricingModels'), {
  loading: () => <ComponentLoader />
});

const GreeksSensitivity = dynamic(() => import('@/components/GreeksSensitivity'), {
  loading: () => <ComponentLoader />
});

const VolatilitySurface = dynamic(() => import('@/components/VolatilitySurface'), {
  loading: () => <ComponentLoader />
});

const PortfolioOptimizer = dynamic(() => import('@/components/PortfolioOptimizer'), {
  loading: () => <ComponentLoader />
});

function ComponentLoader() {
  return (
    <div className="bg-black border border-gray-800 rounded-lg shadow-2xl p-6 backdrop-blur-lg">
      <div className="h-64 flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          <div className="text-gray-400 text-sm font-mono">LOADING COMPONENT...</div>
        </div>
      </div>
    </div>
  );
}
export default function HomePage() {
  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        <Suspense fallback={<ComponentLoader />}>
          <OptionCalculator />
        </Suspense>
        <Suspense fallback={<ComponentLoader />}>
          <DeltaHedgingEngine />
        </Suspense>
      </section>
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        <Suspense fallback={<ComponentLoader />}>
          <GreeksSensitivity />
        </Suspense>
        <Suspense fallback={<ComponentLoader />}>
          <VolatilitySurface />
        </Suspense>
      </section>
      <section>
        <Suspense fallback={<ComponentLoader />}>
          <AdvancedPricingModels />
        </Suspense>
      </section>
      <section className="flex justify-center">
        <div className="w-full max-w-7xl">
          <Suspense fallback={<ComponentLoader />}>
            <PortfolioOptimizer />
          </Suspense>
        </div>
      </section>
    </div>
  );
}
