import { useState, useEffect } from 'react';

// Define the processing steps based on the backend workflow
const steps = [
  { id: 'analyzing', label: 'Analyzing Query', description: 'Understanding your question and creating execution plan' },
  { id: 'executing', label: 'Executing Multi-Tool Plan', description: 'Retrieving documents and web search results' },
  { id: 'evaluating', label: 'Evaluating Documents', description: 'Analyzing document relevance and quality' },
  { id: 'assessing', label: 'Assessing Context', description: 'Checking if context is sufficient for answer' },
  { id: 'generating', label: 'Generating Answer', description: 'Creating comprehensive response from sources' }
];

const ProgressBar = ({ isVisible, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isVisible) {
      setCurrentStep(0);
      setProgress(0);
      return;
    }

    // Simulate the backend processing steps
    const stepDuration = 800; // Duration for each step in ms
    const progressInterval = 50; // Update progress every 50ms
    const maxProgress = 95; // Don't reach 100% until API completes

    let stepIndex = 0;
    let stepProgress = 0;

    const progressTimer = setInterval(() => {
      stepProgress += (100 / (stepDuration / progressInterval));

      if (stepProgress >= 100) {
        stepProgress = 0;
        stepIndex++;

        // Keep cycling through steps but cap at maxProgress
        if (stepIndex >= steps.length) {
          stepIndex = steps.length - 1; // Stay on last step
        } else {
          setCurrentStep(stepIndex);
        }
      }

      const totalProgress = ((stepIndex * 100) + stepProgress) / steps.length;
      setProgress(Math.min(totalProgress, maxProgress));
    }, progressInterval);

    return () => clearInterval(progressTimer);
  }, [isVisible, onComplete]);

  if (!isVisible) return null;

  return (
    <div className="bg-gradient-to-br from-[#111] to-[#0a0a0a] border border-[#222] rounded-3xl p-8 mb-6 shadow-2xl shadow-black/50">
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        <div className="w-1 h-8 bg-gradient-to-b from-[#22c55e] to-transparent rounded-full"></div>
        <h2 className="text-white text-2xl font-bold tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
          Processing Your Question
        </h2>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <span className="text-[#888] text-sm font-medium" style={{fontFamily: 'Product Sans, sans-serif'}}>
            Progress
          </span>
          <span className="text-[#22c55e] text-sm font-bold" style={{fontFamily: 'Product Sans, sans-serif'}}>
            {Math.round(progress)}%
          </span>
        </div>
        
        {/* Progress bar track */}
        <div className="w-full bg-[#1a1a1a] rounded-full h-2 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-[#22c55e] to-[#16a34a] rounded-full transition-all duration-300 ease-out relative"
            style={{ width: `${progress}%` }}
          >
            {/* Animated shimmer effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Current Step */}
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          {/* Step info */}
          <div className="flex-1">
            <div className="text-white text-lg font-semibold mb-1" style={{fontFamily: 'Product Sans, sans-serif'}}>
              {steps[currentStep]?.label}
            </div>
            <div className="text-[#888] text-sm" style={{fontFamily: 'Product Sans, sans-serif'}}>
              {steps[currentStep]?.description}
            </div>
          </div>
        </div>

        {/* Steps list */}
        <div className="mt-6 space-y-3">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center gap-3">
              {/* Step indicator */}
              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                index < currentStep 
                  ? 'bg-[#22c55e] border-[#22c55e]' 
                  : index === currentStep
                  ? 'border-[#22c55e] bg-transparent animate-pulse'
                  : 'border-[#333] bg-transparent'
              }`}>
                {index < currentStep && (
                  <svg className="w-2.5 h-2.5 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
              
              {/* Step label */}
              <span className={`text-sm font-medium ${
                index < currentStep 
                  ? 'text-[#22c55e]' 
                  : index === currentStep
                  ? 'text-white'
                  : 'text-[#666]'
              }`} style={{fontFamily: 'Product Sans, sans-serif'}}>
                {step.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;
