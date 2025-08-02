Financial Simulator
An educational options trading dashboard for learning derivatives pricing, risk management, and portfolio theory.

🔗 Live Demo
Try it now: https://financialsimulator-frontend.onrender.com/

No installation needed - just open the link and start experimenting with options pricing and delta hedging.

Overview
This project simulates real options trading scenarios to help students and finance enthusiasts understand how derivatives work. You can play around with different option parameters, see how the Greeks change, and practice delta hedging without risking actual money.

Features
Options Pricing Calculator
Black-Scholes pricing model implementation

Real-time Greeks calculation (Delta, Gamma, Vega, Theta)

Interactive parameter sliders for spot price, strike, volatility, etc.

Support for both calls and puts

Delta Hedging Simulator
Practice hedging option positions

See how P&L changes as the underlying moves

Auto-rebalancing feature to maintain delta neutrality

Track hedging effectiveness and transaction costs

Advanced Pricing Models
Heston stochastic volatility model

Merton jump diffusion model

Compare results across different pricing approaches

Portfolio Optimization Tools
Mean-variance optimization

Efficient frontier plotting

Sharpe ratio maximization

Risk-return trade-off visualization

Volatility Surface Visualization
surface plots showing implied volatility

Interactive parameter adjustment

Real-time surface updates

Quick Start
Just visit https://financialsimulator-frontend.onrender.com/ and start using it immediately.

Usage Guide
Basic Option Pricing
Enter your option parameters in the left panel

Choose call or put option type

Click "Calculate Option Price"

Watch the Greeks update automatically

Delta Hedging Practice
Add option positions using the "Add Position" button

Turn on auto-rebalancing if desired

Monitor hedge P&L as underlying price moves

Try to maintain delta-neutral portfolio

Volatility Analysis
Adjust volatility slider to see price sensitivity

Higher volatility = higher option prices (especially ATM)

Observe Vega changes across different strikes

Educational Objectives
This simulator helps you understand:

Option Pricing Theory: Black-Scholes formula implementation

The Greeks: Delta, Gamma, Vega, Theta behavior

Risk Management: Delta hedging for market makers

Volatility Impact: How vol affects option pricing

Portfolio Theory: Modern portfolio optimization

Risk-Return Trade-offs: Efficient frontier concepts

Learning Exercises
Exercise 1: Greek Sensitivity
Try changing the spot price and observe Delta behavior. Notice how:

Delta approaches 1.0 for deep ITM calls

Delta approaches 0.0 for deep OTM calls

Exercise 2: Time Decay Analysis
Set up an ATM option and monitor Theta:

Observe accelerating time decay near expiration

Compare ATM vs ITM/OTM time decay rates

Exercise 3: Volatility Impact Study
Compare option prices at different volatility levels:

20% volatility vs 40% volatility

Which options show highest Vega sensitivity?

Exercise 4: Delta Hedging Simulation
Create a short call position and practice hedging:

Monitor P&L changes with underlying movement

Experiment with rebalancing frequencies

License
MIT License - use for learning, teaching, or any purpose.

Disclaimer
Educational use only. This simulator is designed for learning and should not be used for actual trading decisions. The models are simplified and don't account for real-world factors.

Support
Issues: Report bugs via GitHub Issues

Questions: Use GitHub Discussions

Live Demo: https://financialsimulator-frontend.onrender.com/

Built for students, educators, and finance enthusiasts learning quantitative finance and derivatives theory.
