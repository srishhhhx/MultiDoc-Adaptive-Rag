import React, { useState, useEffect } from "react";

const Typewriter = ({ 
  text, 
  speed = 30, 
  className = "",
  onComplete = () => {},
  showCursor = true,
  cursorClassName = ""
}) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    // Reset when text changes
    setDisplayedText("");
    setCurrentIndex(0);
    setIsComplete(false);
  }, [text]);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, speed);

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
      onComplete();
    }
  }, [currentIndex, text, speed, isComplete, onComplete]);

  return (
    <span className={className}>
      {displayedText}
      {showCursor && !isComplete && (
        <span className={`inline-block w-[2px] h-[1em] bg-current ml-[2px] animate-pulse ${cursorClassName}`}>
          |
        </span>
      )}
    </span>
  );
};

export default Typewriter;
