# Messaging Architecture

This document outlines the architecture for handling different message types in MiniMeAI.

## Core Principles

1. **Pull-based Architecture**: All data sources expose an API endpoint for data retrieval
2. **Consistent Metadata**: Standardized metadata across all message types
3. **Unified Processing**: Common processing pipeline for all data types
4. **Decoupled Components**: Services operate independently with minimal dependencies
5. **Centralized Vector Storage**: All processed data is stored in a common vector database

## Email Flow

```
┌─────────────┐        ┌──────────────┐        ┌────────────┐        ┌──────────────┐
│  Email IMAP  │───────▶│ Email Fetcher│───────▶│  Processor │───────▶│  Vector DB   │
└─────────────┘        └──────────────┘        └────────────┘        └──────────────┘
                             │                       ▲
                             │                       │
                             ▼                       │
                        ┌────────────┐               │
                        │ Local Cache │              │
                        └────────────┘               │
                             │                       │
                             │                       │
                             ▼                       │
                        ┌────────────┐               │
                        │ /api/emails │───────────────┘
                        └────────────┘
```

### Process:
1. Email Fetcher connects to IMAP servers
2. New emails are fetched and stored in a local cache
3. Emails are made available via `/api/emails` endpoint
4. Processor pulls emails from this endpoint
5. Processor handles embedding and storage in Vector DB

## WhatsApp Flow

```
┌─────────────┐        ┌───────────────┐        ┌────────────┐        ┌──────────────┐
│ WhatsApp Web│───────▶│WhatsApp Bridge│───────▶│  Processor │───────▶│  Vector DB   │
└─────────────┘        └───────────────┘        └────────────┘        └──────────────┘
                             │                       ▲
                             │                       │
                             ▼                       │
                        ┌────────────┐               │
                        │ Local Cache │              │
                        └────────────┘               │
                             │                       │
                             │                       │
                             ▼                       │
                        ┌────────────┐               │
                        │/api/whatsapp│──────────────┘
                        └────────────┘
```

### Process:
1. WhatsApp Bridge connects to WhatsApp Web
2. New messages are received and stored in a local cache
3. Messages are made available via `/api/whatsapp` endpoint
4. Processor pulls messages from this endpoint
5. Processor handles embedding and storage in Vector DB

## Message Format Standardization

All message types are converted to a common format with these key fields:

| Field       | Description                                   | Example                      |
|-------------|-----------------------------------------------|-----------------------------|
| id          | Unique identifier                            | "msg_12345"                 |
| source_type | Origin of the message                        | "email", "whatsapp"         |
| text        | Main message content                         | "Hello, how are you?"       |
| sender      | Who sent the message                         | "user@example.com"          |
| date        | Human-readable date                          | "2025-03-26T12:34:56.789Z"  |
| timestamp   | Unix timestamp for sorting                   | 1711482896                  |
| metadata    | Additional source-specific information       | {from: "...", to: "..."}    |

This standardization ensures consistent processing across all data sources while preserving source-specific metadata. 