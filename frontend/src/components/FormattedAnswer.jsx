import React from "react";

const FormattedAnswer = ({ text, className = "", showCursor = false }) => {
  // Function to parse and format the text
  const formatText = (text, showCursor) => {
    if (!text) return [];
    
    // Remove all ** markers completely for cleaner display
    let cleanedText = text.replace(/\*\*/g, '');
    // Remove excessive bullet points - only show if there are multiple bullets
    cleanedText = cleanedText.replace(/^\s*\*\s+/gm, '');
    
    // Split by double newlines for paragraphs
    const paragraphs = cleanedText.split(/\n\n+/);
    const formatted = [];
    let key = 0;

    paragraphs.forEach((paragraph, pIndex) => {
      const lines = paragraph.split('\n');
      
      // Check if this is a bulleted list section - only show bullets if 3+ items
      const bulletLines = lines.filter(line => line.match(/^\s*[\*\-•]\s+/));
      const isBulletSection = bulletLines.length >= 3;
      
      const isLastParagraph = pIndex === paragraphs.length - 1;
      
      if (isBulletSection) {
        // Group consecutive bullets together
        const bullets = [];
        lines.forEach((line, lineIndex) => {
          const bulletMatch = line.match(/^\s*[\*\-•]\s+(.+)$/);
          if (bulletMatch) {
            const content = bulletMatch[1];
            const isLastBullet = isLastParagraph && lineIndex === lines.length - 1;
            bullets.push(
              <div key={key++} className="flex gap-3 mb-3 pl-2">
                <span className="text-[#22c55e] mt-1 flex-shrink-0 font-bold">•</span>
                <span className="flex-1 text-[#ccc] leading-relaxed">
                  {formatInlineText(content)}
                  {isLastBullet && showCursor && (
                    <span className="inline-block w-[3px] h-[18px] bg-[#22c55e] ml-[2px] animate-pulse align-text-bottom rounded-sm"></span>
                  )}
                </span>
              </div>
            );
          }
        });
        
        if (bullets.length > 0) {
          formatted.push(
            <div key={key++} className="mb-4 space-y-1">
              {bullets}
            </div>
          );
        }
      } else {
        // Regular paragraph
        lines.forEach((line, lineIndex) => {
          if (line.trim()) {
            const isLastLine = isLastParagraph && lineIndex === lines.length - 1;
            formatted.push(
              <p key={key++} className="mb-5 leading-[1.8] text-[#ccc]">
                {formatInlineText(line)}
                {isLastLine && showCursor && (
                  <span className="inline-block w-[3px] h-[18px] bg-[#22c55e] ml-[2px] animate-pulse align-text-bottom rounded-sm"></span>
                )}
              </p>
            );
          }
        });
      }
      
      // Add spacing between sections
      if (pIndex < paragraphs.length - 1) {
        formatted.push(<div key={key++} className="h-2" />);
      }
    });

    return formatted;
  };

  // Function to format inline text (bold, etc.)
  const formatInlineText = (text) => {
    const parts = [];
    let key = 0;
    
    // Just process the text directly since we removed ** already
    const segments = [text];
    
    segments.forEach((segment) => {
      if (segment) {
        // Regular text - split by colons to highlight labels
        const colonSplit = segment.split(/(:)/g);
        colonSplit.forEach((part, idx) => {
          if (part === ':' && idx > 0) {
            // Check if the previous part looks like a label
            const prevPart = colonSplit[idx - 1];
            if (prevPart && /^[A-Z][a-zA-Z\s&]+$/.test(prevPart.trim())) {
              parts[parts.length - 1] = (
                <strong key={`label-${key++}`} className="font-semibold text-[#e0e0e0]">
                  {prevPart}:
                </strong>
              );
            } else {
              parts.push(<span key={key++}>:</span>);
            }
          } else if (part && part !== ':') {
            parts.push(<span key={key++}>{part}</span>);
          }
        });
      }
    });

    return parts;
  };

  return (
    <div className={className}>
      {formatText(text, showCursor)}
    </div>
  );
};

export default FormattedAnswer;
