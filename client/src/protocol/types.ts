export type MessageType =
  | "handshake"
  | "audio"
  | "text"
  | "control"
  | "metadata"
  | "user_text";

export const VERSIONS_MAP = {
  0: 0b00000000,
} as const;

export const MODELS_MAP = {
  0: 0b00000000,
} as const;

export type VERSION = keyof typeof VERSIONS_MAP;

export type MODEL = keyof typeof MODELS_MAP;

export type WSMessage =
  | {
      type: "handshake";
      version: VERSION;
      model: MODEL;
    }
  | {
      type: "audio";
      data: Uint8Array;
    }
  | {
      type: "text";
      data: string;
      timestamp: number; // Unix timestamp in milliseconds
    }
  | {
      type: "control";
      action: CONTROL_MESSAGE;
    }
  | {
      type: "metadata";
      data: unknown;
    }
  | {
    type: "error";
    data: string;
  }
  | {
    type:"ping";
  }
  | {
    type: "user_text";
    data: string;
    timestamp: number; // Unix timestamp in milliseconds
  }

// Conversation message for unified display
export type ConversationMessage = {
  id: string;
  speaker: "user" | "assistant";
  text: string;
  timestamp: number;
};


export type SocketStatus = "connected" | "disconnected" | "connecting";

export const CONTROL_MESSAGES_MAP = {
  start: 0b00000000,
  endTurn: 0b00000001,
  pause: 0b00000010,
  restart: 0b00000011,
} as const;

export type CONTROL_MESSAGE = keyof typeof CONTROL_MESSAGES_MAP;
