import React, { useState, useEffect } from 'react';

const CircularGauge = ({ 
  value, 
  max = 100, 
  size = 'md', 
  color = 'green',
  label,
  showPercentage = true 
}) => {
  const [animatedValue, setAnimatedValue] = useState(0);
  const percentage = (value / max) * 100;
  const animatedPercentage = (animatedValue / max) * 100;

  // Animate the value on mount or when value changes - slower animation
  useEffect(() => {
    setAnimatedValue(0);
    const duration = 2500; // 2.5 seconds (slower)
    const steps = 80;
    const increment = value / steps;
    const stepDuration = duration / steps;

    let currentStep = 0;
    const timer = setInterval(() => {
      currentStep++;
      if (currentStep <= steps) {
        setAnimatedValue(prev => Math.min(prev + increment, value));
      } else {
        setAnimatedValue(value);
        clearInterval(timer);
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, [value]);
  
  // Size configurations - slightly reduced sizes
  const sizes = {
    sm: { width: 100, stroke: 9, fontSize: 'text-xl', labelSize: 'text-xs', outerStroke: 11 },
    md: { width: 140, stroke: 11, fontSize: 'text-3xl', labelSize: 'text-sm', outerStroke: 13 },
    lg: { width: 170, stroke: 13, fontSize: 'text-4xl', labelSize: 'text-base', outerStroke: 15 }
  };
  
  const config = sizes[size] || sizes.md;
  const radius = (config.width - config.outerStroke * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animatedPercentage / 100) * circumference;
  
  // Color configurations with enhanced gradients
  const colors = {
    green: {
      trail: 'rgba(34, 197, 94, 0.08)',
      path: '#22c55e',
      glow: 'rgba(34, 197, 94, 0.3)',
      text: '#22c55e',
      background: 'rgba(34, 197, 94, 0.03)'
    },
    blue: {
      trail: 'rgba(107, 159, 255, 0.08)',
      path: '#6b9fff',
      glow: 'rgba(107, 159, 255, 0.3)',
      text: '#6b9fff',
      background: 'rgba(107, 159, 255, 0.03)'
    },
    red: {
      trail: 'rgba(255, 107, 107, 0.08)',
      path: '#ff6b6b',
      glow: 'rgba(255, 107, 107, 0.3)',
      text: '#ff6b6b',
      background: 'rgba(255, 107, 107, 0.03)'
    },
    yellow: {
      trail: 'rgba(251, 191, 36, 0.08)',
      path: '#fbbf24',
      glow: 'rgba(251, 191, 36, 0.3)',
      text: '#fbbf24',
      background: 'rgba(251, 191, 36, 0.03)'
    }
  };
  
  const colorConfig = colors[color] || colors.green;
  
  return (
    <div className="flex flex-col items-center gap-6">
      {/* Main gauge container */}
      <div className="relative flex items-center justify-center" style={{ width: config.width, height: config.width }}>
        {/* Background glow */}
        <div 
          className="absolute inset-0 rounded-full"
          style={{ 
            background: `radial-gradient(circle, ${colorConfig.glow} 0%, transparent 70%)`,
            filter: 'blur(40px)',
            opacity: 0.5
          }}
        />
        
        {/* SVG Progress Ring */}
        <svg
          width={config.width}
          height={config.width}
          className="transform -rotate-90 relative z-10"
        >
          <defs>
            {/* Gradient definition */}
            <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={colorConfig.path} />
              <stop offset="100%" stopColor={colorConfig.path} stopOpacity="0.6" />
            </linearGradient>
          </defs>
          
          {/* Background track */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={config.stroke}
            fill="none"
            strokeLinecap="round"
          />
          
          {/* Progress circle with glow */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            stroke={colorConfig.path}
            strokeWidth={config.stroke}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
            style={{
              filter: `drop-shadow(0 0 8px ${colorConfig.path})`,
              opacity: 0.9
            }}
          />
        </svg>
        
        {/* Center percentage display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {showPercentage && (
            <>
              <div 
                className={`${config.fontSize} font-black tracking-tight`}
                style={{ 
                  color: colorConfig.text,
                  lineHeight: 1
                }}
              >
                {animatedPercentage.toFixed(0)}
                <span className="text-[0.35em] opacity-70">%</span>
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* Label */}
      {label && (
        <div className={`${config.labelSize} text-[#aaa] font-semibold text-center tracking-wide`}>
          {label}
        </div>
      )}
    </div>
  );
};

export default CircularGauge;
