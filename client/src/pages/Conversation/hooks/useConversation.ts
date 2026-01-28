import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { ConversationMessage } from "../../../protocol/types";

/**
 * Unified conversation hook that combines both user and assistant messages
 * in chronological order.
 */
export const useConversation = () => {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);

    if (message.type === "text") {
      const newMessage: ConversationMessage = {
        id: `assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        speaker: "assistant",
        text: message.data,
        timestamp: message.timestamp,
      };
      setMessages((prev) => {
        // Merge consecutive assistant messages into one
        if (prev.length > 0) {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg.speaker === "assistant") {
            // Append to existing assistant message
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...lastMsg,
              text: lastMsg.text + message.data,
              timestamp: message.timestamp, // Update timestamp to latest
            };
            return updated;
          }
        }
        return [...prev, newMessage];
      });
    } else if (message.type === "user_text") {
      const newMessage: ConversationMessage = {
        id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        speaker: "user",
        text: message.data,
        timestamp: message.timestamp,
      };
      setMessages((prev) => {
        // For user text, merge consecutive user messages within 5 seconds
        // This helps group ASR chunks that belong to the same utterance
        if (prev.length > 0) {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg.speaker === "user" && (message.timestamp - lastMsg.timestamp) < 5000) {
            // Append to existing user message if within 5 seconds
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...lastMsg,
              text: lastMsg.text + " " + message.data,
              timestamp: message.timestamp,
            };
            return updated;
          }
        }
        return [...prev, newMessage];
      });
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setMessages([]);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket, onSocketMessage]);

  return { messages };
};

