# WhatsApp Integration for MiniMeAI

This module integrates WhatsApp messaging into the MiniMeAI system using the whatsapp-web.js library.

## Architecture

The WhatsApp integration follows a pull-based approach similar to the email service:

1. **WhatsApp Bridge**: A Node.js service that connects to WhatsApp Web and stores messages locally.
2. **Message Cache**: WhatsApp messages are stored in a JSON file and served via an API endpoint.
3. **Processor Integration**: The processor service periodically pulls messages from the WhatsApp bridge.

## Components

### WhatsApp Bridge

The WhatsApp bridge provides the following functionality:

- Connects to WhatsApp Web via QR code authentication
- Receives and stores messages in a local cache
- Exposes an API endpoint (`/api/whatsapp`) to serve messages
- Provides a web interface for QR code scanning and connection status

### API Endpoints

The bridge exposes the following endpoints:

- `GET /`: Web interface for QR code scanning and connection status
- `GET /api/status`: Returns the current connection status
- `GET /api/whatsapp`: Returns cached WhatsApp messages (with optional limit parameter)
- `POST /api/send`: Sends a WhatsApp message

## Configuration

The WhatsApp bridge supports the following environment variables:

- `WHATSAPP_BRIDGE_PORT`: Port for the bridge service (default: `3001`)
- `WHATSAPP_SESSION_DIR`: Directory for storing WhatsApp session data (default: `../wwebjs_auth`)
- `MAX_CACHED_MESSAGES`: Maximum number of messages to keep in cache (default: `1000`)

## Processor Integration

The processor service connects to the WhatsApp bridge to fetch messages, then:

1. Stores messages in ChromaDB with appropriate metadata
2. Tracks processed messages to avoid duplication
3. Provides semantic search capabilities across all message types

## Access from Other Services

To access WhatsApp messages from other services:

```
GET http://whatsapp_bridge:3001/api/whatsapp?limit=50
```

The response will be a JSON object with the following format:

```json
{
  "messages": [
    {
      "id": "message-id",
      "source_type": "whatsapp",
      "text": "Message content",
      "sender": "Sender name",
      "chat": "Chat name",
      "date": "2023-05-01T12:34:56.789Z",
      "timestamp": 1682945696.789,
      "metadata": {
        "source_id": "message-id",
        "from": "12345678901@c.us",
        "hasMedia": false,
        "isGroup": false,
        "isForwarded": false,
        "type": "chat"
      }
    }
  ],
  "total": 100
}
```

## Setup

The WhatsApp bridge is designed to run inside a Docker container as part of the MiniMeAI stack.

### Environment Variables

The following environment variables can be configured in the `.env` file:

- `WHATSAPP_BRIDGE_PORT`: Port for the WhatsApp bridge web interface (default: 3001)
- `WHATSAPP_SESSION_DIR`: Directory to store WhatsApp session data (default: `/app/src/whatsapp/.wwebjs_auth`)
- `PROCESSING_SERVICE_URL`: URL of the processing service (default: `http://processor:8080/api/ingest`)

## Usage

### Starting the Service

The WhatsApp bridge starts automatically when you run the MiniMeAI stack:

```bash
docker-compose up -d
```

### Initial Authentication

1. Open a web browser and navigate to `http://your-raspberry-pi-ip:3001`
2. Scan the QR code with your WhatsApp mobile app
3. Once authenticated, the bridge will start processing messages

### API Endpoints

The WhatsApp bridge provides the following API endpoints:

- `GET /api/status`: Get the current connection status
- `POST /api/send`: Send a WhatsApp message (parameters: `to`, `message`)

Example usage:

```bash
# Get status
curl http://your-raspberry-pi-ip:3001/api/status

# Send message
curl -X POST http://your-raspberry-pi-ip:3001/api/send \
  -H "Content-Type: application/json" \
  -d '{"to": "1234567890@c.us", "message": "Hello from MiniMeAI!"}'
```

## Troubleshooting

### QR Code Authentication Issues

If you have problems scanning the QR code:

1. Try restarting the WhatsApp bridge service: `docker-compose restart whatsapp_bridge`
2. Clear the authentication data if needed: `sudo rm -rf data/whatsapp/.wwebjs_auth`
3. Check the logs: `docker-compose logs whatsapp_bridge`

### Connection Issues

If the bridge cannot connect to WhatsApp:

1. Ensure your Raspberry Pi has a stable internet connection
2. Check the WhatsApp service on your phone is working correctly
3. Verify that the Chrome browser inside the container is working properly

## Development Notes

- The WhatsApp bridge uses Node.js 16 LTS for maximum compatibility
- The headless Chrome browser runs in lightweight mode to suit Raspberry Pi resources
- Messages are processed asynchronously to avoid blocking the main thread 