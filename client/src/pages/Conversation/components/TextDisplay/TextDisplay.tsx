import { FC, useEffect, useRef } from "react";
import { useConversation } from "../../hooks/useConversation";
import { ConversationMessage } from "../../../../protocol/types";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
};

// Format timestamp for display
const formatTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

// Single message component
const MessageBubble: FC<{ message: ConversationMessage; isLatest: boolean }> = ({ message, isLatest }) => {
  const isUser = message.speaker === "user";

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
        } ${isLatest ? 'ring-2 ring-offset-1 ring-opacity-50' : ''} ${
          isLatest && isUser ? 'ring-blue-300' : isLatest ? 'ring-gray-400' : ''
        }`}
      >
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-xs font-semibold ${isUser ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}`}>
            {isUser ? 'You' : 'Assistant'}
          </span>
          <span className={`text-xs ${isUser ? 'text-blue-200' : 'text-gray-400 dark:text-gray-500'}`}>
            {formatTime(message.timestamp)}
          </span>
        </div>
        <div className={`text-sm ${isLatest ? 'font-medium' : ''}`}>
          {message.text}
        </div>
      </div>
    </div>
  );
};

export const TextDisplay: FC<TextDisplayProps> = ({
  containerRef,
}) => {
  const { messages } = useConversation();
  const prevScrollTop = useRef(0);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, containerRef]);

  return (
    <div className="h-full w-full max-w-full max-h-full p-3 space-y-2 overflow-y-auto">
      {messages.length === 0 ? (
        <div className="text-center text-gray-400 dark:text-gray-500 text-sm py-4">
          Start speaking to see the conversation...
        </div>
      ) : (
        messages.map((msg, idx) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            isLatest={idx === messages.length - 1}
          />
        ))
      )}
    </div>
  );
};
