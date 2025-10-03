import React, { useState, useEffect } from "react";

const TypewriterHeading = ({ text, className, style }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
    setIsComplete(false);
  }, [text]);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, 80); // Slower typing speed for heading

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
    }
  }, [currentIndex, text, isComplete]);

  return (
    <h1 className={className} style={style}>
      {displayedText}
      {!isComplete && (
        <span className="animate-pulse text-[#22c55e]">|</span>
      )}
    </h1>
  );
};

export default TypewriterHeading;
