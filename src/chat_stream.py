"""Load chat data, group by chat_id, yield messages chronologically."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Message:
    id: str
    publisher_id: str
    user_id: str
    visitor_id: str
    chat_id: str
    session_id: str
    role: str
    text: str
    created_at: str


@dataclass
class Chat:
    chat_id: str
    messages: list[Message]

    @property
    def visitor_ids(self) -> set[str]:
        return {m.visitor_id for m in self.messages}

    @property
    def session_ids(self) -> set[str]:
        return {m.session_id for m in self.messages}


class ChatStream:
    def __init__(self, csv_path: str | Path):
        self.chats: dict[str, Chat] = {}
        self._load(csv_path)

    def _load(self, csv_path: str | Path):
        messages_by_chat: dict[str, list[Message]] = {}

        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = Message(
                    id=row['id'],
                    publisher_id=row['publisher_id'],
                    user_id=row['user_id'],
                    visitor_id=row['visitor_id'],
                    chat_id=row['chat_id'],
                    session_id=row['session_id'],
                    role=row['role'],
                    text=row['text'],
                    created_at=row['created_at'],
                )
                messages_by_chat.setdefault(msg.chat_id, []).append(msg)

        # Sort messages within each chat by created_at
        for chat_id, messages in messages_by_chat.items():
            messages.sort(key=lambda m: m.created_at)
            self.chats[chat_id] = Chat(chat_id=chat_id, messages=messages)

    def iter_chats(self) -> list[Chat]:
        """Return chats sorted by their earliest message timestamp."""
        return sorted(self.chats.values(), key=lambda c: c.messages[0].created_at)


if __name__ == '__main__':
    stream = ChatStream('data/chat-data-3303.csv')
    print(f"Loaded {len(stream.chats)} chats")
    total_msgs = sum(len(c.messages) for c in stream.chats.values())
    print(f"Total messages: {total_msgs}")

    for chat in stream.iter_chats()[:5]:
        user_msgs = [m for m in chat.messages if m.role == 'user']
        asst_msgs = [m for m in chat.messages if m.role == 'assistant']
        print(f"  Chat {chat.chat_id}: {len(chat.messages)} msgs "
              f"({len(user_msgs)} user, {len(asst_msgs)} assistant), "
              f"{len(chat.visitor_ids)} visitors, {len(chat.session_ids)} sessions")
