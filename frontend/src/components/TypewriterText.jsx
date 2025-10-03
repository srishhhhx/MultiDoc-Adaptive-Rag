import React, { useState, useEffect } from "react";

const TypewriterText = ({ text, className, style, delay = 0 }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
    setIsComplete(false);
    setHasStarted(false);
    
    // Start typing after delay
    const startTimeout = setTimeout(() => {
      setHasStarted(true);
    }, delay);

    return () => clearTimeout(startTimeout);
  }, [text, delay]);

  useEffect(() => {
    if (hasStarted && currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, 50); // Medium typing speed

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
    }
  }, [currentIndex, text, isComplete, hasStarted]);

  return (
    <p className={className} style={style}>
      {displayedText}
      {hasStarted && !isComplete && (
        <span className="animate-pulse text-[#666]">|</span>
      )}
    </p>
  );
};

export default TypewriterText;
