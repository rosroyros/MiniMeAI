// MiniMeAI Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Constants
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('query');
    const loadingIndicator = document.getElementById('loading');
    const chatMessages = document.getElementById('chatMessages');
    const newConversationBtn = document.getElementById('newConversation');
    const exampleQueries = document.querySelectorAll('.example-query');
    const refreshHealthBtn = document.getElementById('refreshHealthBtn');
    
    // Store conversation history
    let conversationHistory = [];
    
    // Initialize from session storage if available
    try {
        const savedHistory = sessionStorage.getItem('conversationHistory');
        if (savedHistory) {
            conversationHistory = JSON.parse(savedHistory);
            renderConversationHistory();
        }
    } catch (e) {
        console.error('Error loading conversation history:', e);
    }
    
    // Initialize health dashboard
    initializeHealthDashboard();
    
    // Handle form submission with AJAX
    if (queryForm) {
        queryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const query = queryInput.value.trim();
            if (!query) {
                return;
            }
            
            // Add user message to chat
            addUserMessage(query);
            
            // Clear input
            queryInput.value = '';
            
            // Show loading indicator
            showLoading(true);
            
            // Send AJAX request
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    'query': query,
                    'conversation_history': conversationHistory
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                showLoading(false);
                
                if (data.error) {
                    addAIMessage('Sorry, I encountered an error: ' + data.error);
                } else if (data.response) {
                    // Add AI message to chat
                    addAIMessage(data.response);
                } else {
                    addAIMessage('Sorry, I didn\'t get a response from the server.');
                }
            })
            .catch(error => {
                showLoading(false);
                addAIMessage('Sorry, there was an error processing your request: ' + error.message);
            });
        });
    }
    
    // Handle new conversation button
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Confirm before clearing
            if (conversationHistory.length > 0) {
                if (!confirm('Start a new conversation? This will clear the current chat history.')) {
                    return;
                }
            }
            
            // Clear conversation history
            conversationHistory = [];
            sessionStorage.removeItem('conversationHistory');
            
            // Clear chat UI except for welcome message
            chatMessages.innerHTML = '';
            addAIMessage('Hello! I\'m MiniMeAI, your personal assistant. Ask me anything about your emails or messages.');
            
            // Focus on input
            queryInput.focus();
        });
    }
    
    // Handle example query clicks
    if (exampleQueries) {
        exampleQueries.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const exampleText = this.textContent;
                queryInput.value = exampleText;
                queryInput.focus();
            });
        });
    }
    
    // Handle refresh health button
    if (refreshHealthBtn) {
        refreshHealthBtn.addEventListener('click', function(e) {
            e.preventDefault();
            updateHealthDashboard();
        });
    }
    
    // Helper function to add user message to chat
    function addUserMessage(message) {
        const messageHtml = `
            <div class="message user-message">
                <div class="message-content">
                    ${formatMessageText(escapeHtml(message))}
                </div>
                <div class="message-info">
                    <small class="text-muted">You</small>
                </div>
            </div>
        `;
        
        appendMessageToChat(messageHtml);
        
        // Add to conversation history
        conversationHistory.push({
            role: 'user',
            content: message
        });
        
        // Save to session storage
        saveConversationHistory();
    }
    
    // Helper function to add AI message to chat
    function addAIMessage(message) {
        const messageHtml = `
            <div class="message ai-message">
                <div class="message-content">
                    ${formatMessageText(message)}
                </div>
                <div class="message-info">
                    <small class="text-muted">MiniMeAI</small>
                </div>
            </div>
        `;
        
        appendMessageToChat(messageHtml);
        
        // Add to conversation history
        conversationHistory.push({
            role: 'assistant',
            content: message
        });
        
        // Save to session storage
        saveConversationHistory();
    }
    
    // Helper function to append message to chat and scroll to bottom
    function appendMessageToChat(messageHtml) {
        // Add message to chat
        chatMessages.insertAdjacentHTML('beforeend', messageHtml);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Helper function to render conversation history
    function renderConversationHistory() {
        // Clear chat
        chatMessages.innerHTML = '';
        
        // Render each message
        conversationHistory.forEach(message => {
            if (message.role === 'user') {
                const messageHtml = `
                    <div class="message user-message">
                        <div class="message-content">
                            ${formatMessageText(escapeHtml(message.content))}
                        </div>
                        <div class="message-info">
                            <small class="text-muted">You</small>
                        </div>
                    </div>
                `;
                appendMessageToChat(messageHtml);
            } else {
                const messageHtml = `
                    <div class="message ai-message">
                        <div class="message-content">
                            ${formatMessageText(message.content)}
                        </div>
                        <div class="message-info">
                            <small class="text-muted">MiniMeAI</small>
                        </div>
                    </div>
                `;
                appendMessageToChat(messageHtml);
            }
        });
    }
    
    // Helper function to save conversation history to session storage
    function saveConversationHistory() {
        try {
            sessionStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
        } catch (e) {
            console.error('Error saving conversation history:', e);
        }
    }
    
    // Helper function to format message text with paragraphs
    function formatMessageText(text) {
        // Convert newlines to paragraphs
        return text.split('\n\n').map(para => 
            `<p>${para.split('\n').join('<br>')}</p>`
        ).join('');
    }
    
    // Helper function to show/hide loading indicator
    function showLoading(show) {
        if (loadingIndicator) {
            loadingIndicator.classList.toggle('d-none', !show);
        }
    }
    
    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Health Dashboard Functions
    
    function initializeHealthDashboard() {
        // Initial update without setting up auto-refresh
        updateHealthDashboard();
    }
    
    function updateHealthDashboard() {
        // Show loading state
        document.getElementById('lastUpdatedTime').textContent = 'Updating...';
        
        // Fetch health data
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                // Update last updated time
                const lastUpdatedEl = document.getElementById('lastUpdatedTime');
                if (lastUpdatedEl) {
                    const now = new Date();
                    lastUpdatedEl.textContent = now.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
                }
                
                // Update email metrics
                if (data.data_sources && data.data_sources.email) {
                    const emailData = data.data_sources.email;
                    
                    // Update count
                    const emailCountEl = document.getElementById('emailCount');
                    if (emailCountEl) {
                        emailCountEl.textContent = emailData.last_24h_count || 0;
                    }
                    
                    // Update timestamp
                    const emailTimestampEl = document.getElementById('emailTimestamp');
                    if (emailTimestampEl && emailData.latest_date) {
                        const date = new Date(emailData.latest_date);
                        // Format as DD/MM
                        const shortDate = `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}`;
                        emailTimestampEl.textContent = shortDate;
                        
                        // Add full timestamp as tooltip
                        const fullTimestamp = date.toLocaleString();
                        emailTimestampEl.parentElement.parentElement.title = fullTimestamp;
                    } else if (emailTimestampEl) {
                        emailTimestampEl.textContent = '-';
                    }
                }
                
                // Update WhatsApp metrics
                if (data.data_sources && data.data_sources.whatsapp) {
                    const whatsappData = data.data_sources.whatsapp;
                    
                    // Update count
                    const whatsappCountEl = document.getElementById('whatsappCount');
                    if (whatsappCountEl) {
                        whatsappCountEl.textContent = whatsappData.last_24h_count || 0;
                    }
                    
                    // Update timestamp
                    const whatsappTimestampEl = document.getElementById('whatsappTimestamp');
                    if (whatsappTimestampEl && whatsappData.latest_date) {
                        const date = new Date(whatsappData.latest_date);
                        // Format as DD/MM
                        const shortDate = `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}`;
                        whatsappTimestampEl.textContent = shortDate;
                        
                        // Add full timestamp as tooltip
                        const fullTimestamp = date.toLocaleString();
                        whatsappTimestampEl.parentElement.parentElement.title = fullTimestamp;
                    } else if (whatsappTimestampEl) {
                        whatsappTimestampEl.textContent = '-';
                    }
                }
                
                // Update Vector DB metrics
                if (data.data_sources && data.data_sources.vector_db) {
                    const vectorDbData = data.data_sources.vector_db;
                    
                    // Update count
                    const vectorDbCountEl = document.getElementById('vectorDbCount');
                    if (vectorDbCountEl) {
                        vectorDbCountEl.textContent = vectorDbData.total_count || 0;
                    }
                    
                    // Update status
                    const vectorDbStatusEl = document.getElementById('vectorDbStatus');
                    if (vectorDbStatusEl) {
                        const status = vectorDbData.status || 'unknown';
                        vectorDbStatusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
                        
                        // Update badge class
                        vectorDbStatusEl.className = 'badge ' + 
                            (status === 'ok' ? 'bg-success' : 
                             status === 'error' ? 'bg-danger' : 'bg-secondary');
                    }
                }
            })
            .catch(error => {
                console.error('Error updating health dashboard:', error);
                
                // Update with error state
                document.getElementById('lastUpdatedTime').textContent = 'Failed to update';
            });
    }
});
