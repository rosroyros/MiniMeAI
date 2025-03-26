// WhatsApp bridge for MiniMeAI using whatsapp-web.js
const { Client, LocalAuth } = require('whatsapp-web.js');
const express = require('express');
const http = require('http');
const ws = require('ws');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

// Create directories for logs if they don't exist
const LOGS_DIR = path.join(__dirname, '../logs');
if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
}

// Create directory for message storage
const MESSAGE_CACHE_PATH = path.join(LOGS_DIR, 'whatsapp_messages.json');

// Configuration from environment variables
const PORT = process.env.WHATSAPP_BRIDGE_PORT || 3001;
// We no longer need this as we won't be pushing messages
// const PROCESSING_SERVICE_URL = process.env.PROCESSING_SERVICE_URL || 'http://api_service:5000/api/ingest';
const SESSION_DIR = process.env.WHATSAPP_SESSION_DIR || path.join(__dirname, '../.wwebjs_auth');
const MAX_CACHED_MESSAGES = parseInt(process.env.MAX_CACHED_MESSAGES || '1000');

console.log(`Starting WhatsApp bridge with Node.js ${process.version}`);
console.log(`Session directory: ${SESSION_DIR}`);
console.log(`Message cache path: ${MESSAGE_CACHE_PATH}`);

// Create directory for session data
if (!fs.existsSync(SESSION_DIR)) {
    fs.mkdirSync(SESSION_DIR, { recursive: true });
}

// Message cache
let messageCache = [];

// Load message cache from disk
function loadMessageCache() {
    try {
        if (fs.existsSync(MESSAGE_CACHE_PATH)) {
            const data = fs.readFileSync(MESSAGE_CACHE_PATH, 'utf8');
            messageCache = JSON.parse(data);
            console.log(`Loaded ${messageCache.length} messages from cache`);
        } else {
            console.log('No message cache found, starting with empty cache');
            messageCache = [];
        }
    } catch (error) {
        console.error('Error loading message cache:', error.message);
        messageCache = [];
    }
}

// Save message cache to disk
function saveMessageCache() {
    try {
        fs.writeFileSync(MESSAGE_CACHE_PATH, JSON.stringify(messageCache, null, 2));
        console.log(`Saved ${messageCache.length} messages to cache`);
    } catch (error) {
        console.error('Error saving message cache:', error.message);
    }
}

// Add a message to the cache
function addMessageToCache(message) {
    messageCache.push(message);
    
    // Limit cache size
    if (messageCache.length > MAX_CACHED_MESSAGES) {
        messageCache = messageCache.slice(-MAX_CACHED_MESSAGES);
    }
    
    // Save to disk after each new message
    saveMessageCache();
}

// Configure whatsapp-web.js client
const client = new Client({
    authStrategy: new LocalAuth({ dataPath: SESSION_DIR }),
    puppeteer: {
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu'
        ]
    }
});

// Create Express app and WebSocket server
const app = express();
const server = http.createServer(app);
const wss = new ws.Server({ server });

// Store connection state
let qr = null;
let connectionState = 'disconnected';

// Handle WhatsApp client events
client.on('qr', (qrCode) => {
    console.log('QR Code received');
    qr = qrCode;
    connectionState = 'qr';
    broadcast({ type: 'qr', qr: qrCode });
    
    // Log QR code to file
    fs.writeFileSync(path.join(LOGS_DIR, 'latest_qr.txt'), qrCode);
});

client.on('ready', () => {
    console.log('WhatsApp client is ready!');
    connectionState = 'connected';
    broadcast({ type: 'status', status: 'connected' });
});

client.on('authenticated', () => {
    console.log('WhatsApp client authenticated');
    qr = null;
});

client.on('auth_failure', (msg) => {
    console.error('Authentication failure:', msg);
    connectionState = 'auth_failed';
    broadcast({ type: 'status', status: 'auth_failed', message: msg });
});

client.on('disconnected', (reason) => {
    console.log('WhatsApp client disconnected:', reason);
    connectionState = 'disconnected';
    broadcast({ type: 'status', status: 'disconnected', reason });
    
    // Attempt to restart client after disconnect
    setTimeout(() => {
        console.log('Attempting to restart WhatsApp client...');
        client.initialize();
    }, 5000);
});

// Track all incoming messages
client.on('message', async (message) => {
    console.log(`New message from ${message.from}: ${message.body}`);
    
    // Create standardized message object
    const messageData = {
        id: message.id._serialized,
        source_type: 'whatsapp',
        text: message.body,
        sender: message.author || message.from,
        chat: message.from.includes('@g.us') ? 'Group Chat' : 'Direct Message',
        date: new Date().toISOString(),
        timestamp: Date.now() / 1000, // Unix timestamp in seconds for consistency with API
        metadata: {
            source_id: message.id._serialized,
            from: message.from,
            hasMedia: message.hasMedia,
            isGroup: message.isGroup,
            isForwarded: message.isForwarded,
            type: message.type
        }
    };
    
    // Get chat info for better context
    try {
        const chat = await message.getChat();
        if (chat.name) {
            messageData.chat = chat.name;
        }
    } catch (error) {
        console.error('Error getting chat info:', error.message);
    }
    
    // Log the message locally
    fs.appendFileSync(
        path.join(LOGS_DIR, 'message_log.json'), 
        JSON.stringify(messageData) + '\n'
    );
    
    // Add to message cache
    addMessageToCache(messageData);
});

// Broadcast to all connected WebSocket clients
const broadcast = (message) => {
    wss.clients.forEach((client) => {
        if (client.readyState === ws.OPEN) {
            client.send(JSON.stringify(message));
        }
    });
};

// API endpoints
app.use(express.json());

// Send message through WhatsApp
app.post('/api/send', async (req, res) => {
    try {
        const { to, message } = req.body;
        
        if (!to || !message) {
            return res.status(400).json({ error: 'Missing required parameters' });
        }
        
        await client.sendMessage(to, message);
        res.json({ success: true });
    } catch (error) {
        console.error('Error sending message:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get WhatsApp messages (NEW ENDPOINT)
app.get('/api/whatsapp', (req, res) => {
    try {
        console.log(`API request received for /api/whatsapp with params: ${JSON.stringify(req.query)}`);
        const limit = parseInt(req.query.limit) || 50;
        // Sort by timestamp (newest first)
        const sortedMessages = [...messageCache].sort((a, b) => 
            (b.timestamp || 0) - (a.timestamp || 0)
        );
        console.log(`Returning ${Math.min(sortedMessages.length, limit)} messages out of ${messageCache.length} total`);
        res.json({ 
            messages: sortedMessages.slice(0, limit),
            total: messageCache.length
        });
    } catch (error) {
        console.error('Error serving messages:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get connection status
app.get('/api/status', (req, res) => {
    res.json({
        status: connectionState,
        authenticated: connectionState === 'connected',
        message_count: messageCache.length
    });
});

// Serve QR code page
app.get('/', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MiniMeAI WhatsApp Bridge</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            #qrcode { margin: 20px auto; }
            #status { margin: 20px 0; padding: 10px; border-radius: 5px; }
            .disconnected { background-color: #ffcccc; }
            .connecting { background-color: #ffffcc; }
            .connected { background-color: #ccffcc; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>MiniMeAI WhatsApp Bridge</h1>
        <div id="status" class="disconnected">Status: Initializing...</div>
        <div id="qrcode"></div>
        
        <script src="https://cdn.jsdelivr.net/npm/qrcode@1.5.0/build/qrcode.min.js"></script>
        <script>
            const statusDiv = document.getElementById('status');
            const qrcodeDiv = document.getElementById('qrcode');
            
            const ws = new WebSocket('ws://' + window.location.host);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'qr') {
                    statusDiv.textContent = 'Status: Please scan QR code with WhatsApp on your phone';
                    statusDiv.className = 'connecting';
                    qrcodeDiv.innerHTML = '';
                    QRCode.toCanvas(document.createElement('canvas'), data.qr, function (error, canvas) {
                        if (error) console.error(error);
                        qrcodeDiv.appendChild(canvas);
                    });
                } else if (data.type === 'status') {
                    if (data.status === 'connected') {
                        statusDiv.textContent = 'Status: Connected to WhatsApp';
                        statusDiv.className = 'connected';
                        qrcodeDiv.innerHTML = '<p>Successfully connected! The bridge is now tracking messages.</p>';
                    } else if (data.status === 'reconnecting' || data.status === 'connecting') {
                        statusDiv.textContent = 'Status: Reconnecting...';
                        statusDiv.className = 'connecting';
                    } else if (data.status === 'disconnected') {
                        statusDiv.textContent = 'Status: Disconnected from WhatsApp' + 
                                              (data.reason ? ' (' + data.reason + ')' : '');
                        statusDiv.className = 'disconnected';
                    } else if (data.status === 'auth_failed') {
                        statusDiv.textContent = 'Status: Authentication failed. Please try again.';
                        statusDiv.className = 'disconnected';
                    }
                }
            };
            
            ws.onopen = function() {
                statusDiv.textContent = 'Status: WebSocket connected, waiting for WhatsApp...';
            };
            
            ws.onclose = function() {
                statusDiv.textContent = 'Status: WebSocket disconnected';
                statusDiv.className = 'disconnected';
            };
        </script>
    </body>
    </html>
  `);
});

// Setup WebSocket connection
wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    
    if (qr && connectionState === 'qr') {
        ws.send(JSON.stringify({ type: 'qr', qr }));
    } else {
        ws.send(JSON.stringify({ type: 'status', status: connectionState }));
    }
});

// Load message cache on startup
loadMessageCache();

// Start server
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
    
    // Initialize WhatsApp client
    client.initialize();
}); 