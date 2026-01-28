import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";

/**
 * Hook to handle user text messages (transcriptions) from the server.
 * Similar to useServerText but for user speech transcripts (message type 0x07).
 */
export const useUserText = () => {
  const [text, setText] = useState<string[]>([]);
  const [totalTextMessages, setTotalTextMessages] = useState(0);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "user_text") {
      setText(text => [...text, message.data]);
      setTotalTextMessages(count => count + 1);
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setText([]);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { text, totalTextMessages };
};

