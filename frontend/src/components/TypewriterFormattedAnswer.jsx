import React, { useState, useEffect } from "react";
import FormattedAnswer from "./FormattedAnswer";

const TypewriterFormattedAnswer = ({ text }) => {
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
      }, 8); // Typing speed - increased pace

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
    }
  }, [currentIndex, text, isComplete]);

  return <FormattedAnswer text={displayedText} showCursor={!isComplete} />;
};

export default TypewriterFormattedAnswer;
