import React, { useState, useRef, useEffect } from 'react';
import '../styles/ChatInterface.css';
import ELogo from '../assets/ELogo.png';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatMessagesRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: "smooth", 
        block: "end" 
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);
    scrollToBottom();

    // Add user message
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage
    }]);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        body: JSON.stringify({ message: userMessage })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Add bot response
      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.response || 'Sorry, I could not process that request.'
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Sorry, there was an error processing your request.'
      }]);
    }

    setIsLoading(false);
  };

  const formatHistoryMessage = (content) => {
    if (!content.startsWith('Conversation History:')) {
      return content;
    }

    const lines = content.split('\n');
    return (
      <div className="history-display">
        <h3>{lines[0]}</h3>
        {lines.slice(2).map((line, index) => {
          if (line.startsWith('   A: ')) {
            return <div key={index} className="history-answer">{line.substring(5)}</div>;
          } else if (line.startsWith('\n')) {
            return <br key={index} />;
          } else if (line.match(/^\d+\. Q: /)) {
            return <div key={index} className="history-question">{line.substring(line.indexOf('Q: ') + 3)}</div>;
          }
          return null;
        })}
      </div>
    );
  };

  // Function to process markdown-style formatting
  const processText = (text) => {
    if (!text) return '';

    // Split text into blocks (tables and non-tables)
    const blocks = text.toString().split(/(\n(?:\|[^\n]+\n)+)/g);
    
    return blocks.map((block, blockIndex) => {
      if (!block) return null;

      // Check if this block is a table
      if (block.trim().startsWith('|')) {
        try {
          const rows = block.trim().split('\n').filter(row => row.trim());
          
          // Filter out separator rows (containing only dashes and pipes)
          const dataRows = rows.filter(row => !row.replace(/[\|\-\s]/g, '').match(/^$/));
          
          if (dataRows.length < 1) return block;

          // Process all rows to get cells
          const processedRows = dataRows.map(row => 
            row.split('|')
              .filter(cell => cell !== undefined)
              .map(cell => cell.trim())
              .filter(cell => cell.length > 0)
          );

          // Calculate the maximum number of columns
          const columnCount = Math.max(...processedRows.map(row => row.length));
          
          // Only create matrix if we have valid data
          if (columnCount > 0) {
            return (
              <div className="matrix-table" key={`table-${blockIndex}`}>
                <div 
                  className="matrix-table-inner"
                  style={{
                    gridTemplateColumns: `repeat(${columnCount}, minmax(120px, 1fr))`
                  }}
                >
                  {processedRows.map((row, rowIndex) => 
                    row.map((cell, cellIndex) => (
                      <div
                        key={`cell-${rowIndex}-${cellIndex}`}
                        className={rowIndex === 0 ? 'matrix-header' : 'matrix-cell'}
                        style={{
                          gridRow: rowIndex + 1,
                          gridColumn: cellIndex + 1
                        }}
                      >
                        {processInlineFormatting(cell)}
                      </div>
                    ))
                  )}
                </div>
              </div>
            );
          }
        } catch (error) {
          console.error('Error processing table:', error);
          return block;
        }
      }

      // For non-table blocks, process as before
      return processInlineFormatting(block);
    }).filter(Boolean);
  };

  // Helper function to process bold text and emojis within table cells
  const processInlineFormatting = (text) => {
    if (!text) return '';
    
    try {
      // First process bold text (text between double asterisks)
      const parts = text.toString().split(/(\*\*[^*]+?\*\*:?)/g);
      
      return parts.map((part, index) => {
        if (!part) return null;
        
        if (part.startsWith('**') && (part.endsWith('**') || part.endsWith('**:'))) {
          const hasColon = part.endsWith('**:');
          const boldText = part.slice(2, hasColon ? -3 : -2);
          return (
            <strong key={`bold-${index}`} className={`bold-text${hasColon ? ' with-colon' : ''}`}>
              {boldText}
            </strong>
          );
        }
        
        // Process emojis in non-bold text
        return part.split(/(\p{Emoji}+)/gu).map((subPart, subIndex) => {
          if (!subPart) return null;
          if (subPart.match(/\p{Emoji}+/gu)) {
            return <span key={`emoji-${index}-${subIndex}`} className="emoji">{subPart}</span>;
          }
          return subPart;
        }).filter(Boolean);
      }).filter(Boolean).flat();
    } catch (error) {
      console.error('Error processing text formatting:', error);
      return text; // Return original text if processing fails
    }
  };

  const formatMessage = (content) => {
    if (content.startsWith('Conversation History:')) {
      return formatHistoryMessage(content);
    }

    // Extract the context label and message content
    const matches = content.match(/^\[(.*?)\]\s*([\s\S]*)$/);
    if (!matches) return processText(content);

    const [_, contextLabel, messageContent] = matches;

    // Split content into paragraphs
    const paragraphs = messageContent.split('\n\n');
    
    return (
      <div className="message-wrapper">
        <div className="context-label">[{contextLabel}]</div>
        <div className="message-text">
          {paragraphs.map((paragraph, pIndex) => {
            // Check if this paragraph contains bullet points
            if (paragraph.split('\n').every(line => line.trim().startsWith('*'))) {
              // This is a bullet point list
              return (
                <ul key={`list-${pIndex}`} className="bullet-list">
                  {paragraph.split('\n').map((line, index) => {
                    // Remove only the first asterisk and any following whitespace
                    const lineContent = line.replace(/^\*\s*/, '');
                    return (
                      <li key={`item-${index}`} className="bullet-item">
                        {processText(lineContent)}
                      </li>
                    );
                  })}
                </ul>
              );
            } else {
              // Regular paragraph
              return (
                <p key={`para-${pIndex}`} className="message-paragraph">
                  {processText(paragraph)}
                </p>
              );
            }
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="chat-container">
      <div className="main-content">
        <div className="chat-interface">
          <div className="chat-header">
            <a href="https://eoxs.com/" target="_blank" rel="noopener noreferrer" className="logo-link">
              <img src={ELogo} alt="EOXS Logo" className="e-logo" />
            </a>
            <div className="header-content">
              <h1>KYNEX</h1>
            </div>
          </div>

          <div className="chat-messages" ref={chatMessagesRef}>
            <div className="messages-container">
              {messages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  <div className="message-content">
                    {formatMessage(message.content)}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message bot">
                  <div className="message-content">
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} className="scroll-anchor" />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="chat-input">
            <div className="chat-input-inner">
              <textarea
                value={inputMessage}
                onChange={(e) => {
                  setInputMessage(e.target.value);
                  // Auto-resize the textarea
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Type your message here..."
                disabled={isLoading}
                rows="1"
                className="chat-input-textarea"
              />
              <button type="submit" disabled={isLoading} className="send-button" aria-label="Send message">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M2 12L20 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M14 6L20 12L14 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 