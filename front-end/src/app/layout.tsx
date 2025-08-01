import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })
export const metadata: Metadata = {
  title: 'Financial Options Trading Dashboard',
  description: 'Options pricing and risk management platform',
}
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-black text-white min-h-screen`}>
        <div className="container mx-auto px-4 py-8">
          <header className="mb-8">
            <h1 className="text-4xl font-bold text-center bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              FINANCIAL OPTIONS TRADING DASHBOARD
            </h1>
            <p className="text-center text-gray-400 mt-2 font-mono text-sm">
              Advanced Pricing Models • Fin. Risk Management • Real-Time Analytics
            </p>
          </header>
          {children}
        </div>
      </body>
    </html>
  )
}
