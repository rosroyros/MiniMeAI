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
    
    console.log("MiniMeAI client initialized");
    
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
            
            console.log("Sending query to server:", query);
            
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
                console.log("Response received:", data);
                
                if (data.error) {
                    addAIMessage('Sorry, I encountered an error: ' + data.error);
                } else if (data.response) {
                    // Add AI message to chat with citations if available
                    if (data.citations && data.citations.length > 0) {
                        console.log("Citations received:", data.citations);
                        console.log("Citation types:", data.citations.map(c => c.source_type));
                        console.log("Citation metadata:", data.citations.map(c => c.metadata));
                        console.log("Citation count:", data.citations.length);
                        console.log("First citation:", JSON.stringify(data.citations[0]));
                        addAIMessageWithCitations(data.response, data.citations);
                    } else {
                        console.log("No citations received, data keys:", Object.keys(data));
                        console.log("Full response object:", JSON.stringify(data));
                        addAIMessage(data.response);
                    }
                } else {
                    addAIMessage('Sorry, I didn\'t get a response from the server.');
                }
            })
            .catch(error => {
                showLoading(false);
                console.error("Error processing request:", error);
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
    
    // Helper function to add AI message with citations to chat
    function addAIMessageWithCitations(message, citations) {
        console.log("Adding AI message with citations. Citation count:", citations.length);
        
        // Format the citations HTML
        const citationsHtml = formatCitations(citations);
        console.log("Formatted citations HTML length:", citationsHtml.length);
        console.log("Citations HTML (first 100 chars):", citationsHtml.substring(0, 100));
        
        const messageHtml = `
            <div class="message ai-message">
                <div class="message-content">
                    ${formatMessageText(message)}
                    <div class="citations-container">
                        <div class="citations-header" onclick="toggleCitations(this)">
                            <i class="bi bi-info-circle me-2"></i>
                            <span>Sources (${citations.length})</span>
                            <i class="bi bi-chevron-down ms-auto"></i>
                        </div>
                        <div class="citations-content">
                            ${citationsHtml}
                        </div>
                    </div>
                </div>
                <div class="message-info">
                    <small class="text-muted">MiniMeAI</small>
                </div>
            </div>
        `;
        
        appendMessageToChat(messageHtml);
        console.log("AI message with citations added to chat");
        
        // Add to conversation history (without citations)
        conversationHistory.push({
            role: 'assistant',
            content: message
        });
        
        // Save to session storage
        saveConversationHistory();
    }
    
    // Helper function to format citations
    function formatCitations(citations) {
        console.log("Formatting citations:", citations);
        
        if (!Array.isArray(citations)) {
            console.error("Citations is not an array:", citations);
            return "Error formatting citations";
        }
        
        return citations.map((citation, index) => {
            const metadata = citation.metadata || {};
            let sourceType = citation.source_type || 'message';
            console.log(`Citation ${index}: type=${sourceType}, metadata=`, metadata);
            
            // Detect WhatsApp messages even if they're classified as "message"
            // Check for WhatsApp indicators in metadata or content
            if (sourceType === 'message' && 
                ((metadata.from && metadata.from.toLowerCase().includes('whatsapp')) ||
                 (metadata.sender && metadata.sender.toLowerCase().includes('whatsapp')) ||
                 (metadata.chat && metadata.chat.toLowerCase().includes('whatsapp')) ||
                 (citation.snippet && citation.snippet.toLowerCase().includes('whatsapp')))) {
                sourceType = 'whatsapp';
                console.log(`Reclassified citation ${index} as WhatsApp based on content`);
            }
            
            // Get source type-specific icon
            let sourceIcon = '';
            if (sourceType === 'email') {
                sourceIcon = '<i class="bi bi-envelope-fill"></i>';
            } else if (sourceType === 'whatsapp') {
                sourceIcon = '<i class="bi bi-whatsapp"></i>';
            } else {
                sourceIcon = '<i class="bi bi-chat-text-fill"></i>';
            }
            
            // Format metadata based on source type
            let metadataHtml = '';
            if (sourceType === 'email') {
                metadataHtml = `
                    <div><strong>From:</strong> ${escapeHtml(metadata.from || 'Unknown')}</div>
                    <div><strong>Subject:</strong> ${escapeHtml(metadata.subject || 'No Subject')}</div>
                    <div><strong>Date:</strong> ${escapeHtml(metadata.date || 'Unknown')}</div>
                `;
            } else if (sourceType === 'whatsapp') {
                metadataHtml = `
                    <div><strong>From:</strong> ${escapeHtml(metadata.sender || 'Unknown')}</div>
                    <div><strong>Chat:</strong> ${escapeHtml(metadata.chat || 'Unknown')}</div>
                    <div><strong>Date:</strong> ${escapeHtml(metadata.date || 'Unknown')}</div>
                `;
            } else {
                metadataHtml = `
                    <div><strong>From:</strong> ${escapeHtml(metadata.sender || metadata.from || 'Unknown')}</div>
                    <div><strong>Subject:</strong> ${escapeHtml(metadata.subject || 'No Subject')}</div>
                    <div><strong>Date:</strong> ${escapeHtml(metadata.date || 'Unknown')}</div>
                `;
            }
            
            // Convert relevance score to percentage
            const relevancePercentage = Math.round((citation.relevance_score || 0) * 100);
            
            return `
            <div class="citation-item" data-source-type="${sourceType}">
                <div class="citation-header">
                    <div class="citation-source-icon">${sourceIcon}</div>
                    <div class="citation-type">${sourceType.charAt(0).toUpperCase() + sourceType.slice(1)}</div>
                    <div class="citation-score">
                        <div class="relevance-indicator" style="width: ${relevancePercentage}%"></div>
                        <span>${relevancePercentage}% relevance</span>
                    </div>
                </div>
                <div class="citation-metadata">
                    ${metadataHtml}
                </div>
                <div class="citation-snippet">
                    ${escapeHtml(citation.snippet || '')}
                </div>
            </div>
            `;
        }).join('');
    }
    
    // Helper function to append message HTML to chat
    function appendMessageToChat(messageHtml) {
        console.log("Appending message to chat:", messageHtml.substring(0, 100) + "...");
        chatMessages.innerHTML += messageHtml;
        
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
    
    // Helper function to escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Show/hide loading indicator
    function showLoading(show) {
        if (loadingIndicator) {
            if (show) {
                loadingIndicator.classList.remove('d-none');
            } else {
                loadingIndicator.classList.add('d-none');
            }
        }
    }
    
    // Initialize health dashboard
    function initializeHealthDashboard() {
        updateHealthDashboard();
    }
    
    // Update health dashboard data
    function updateHealthDashboard() {
        // Show loading state
        document.getElementById('emailCount').textContent = '...';
        document.getElementById('whatsappCount').textContent = '...';
        document.getElementById('vectorDbCount').textContent = '...';
        document.getElementById('lastUpdatedTime').textContent = '...';
        
        // Get health data
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                // Update the health dashboard with the data
                document.getElementById('emailCount').textContent = data.email_count || '0';
                document.getElementById('emailTimestamp').textContent = data.last_email_time || 'None';
                
                document.getElementById('whatsappCount').textContent = data.whatsapp_count || '0';
                document.getElementById('whatsappTimestamp').textContent = data.last_whatsapp_time || 'None';
                
                document.getElementById('vectorDbCount').textContent = data.vector_count || '0';
                
                // Update vector DB status
                const vectorDbStatus = document.getElementById('vectorDbStatus');
                if (data.vector_status === 'ok') {
                    vectorDbStatus.textContent = 'Healthy';
                    vectorDbStatus.className = 'status-badge badge bg-success';
                } else {
                    vectorDbStatus.textContent = 'Error';
                    vectorDbStatus.className = 'status-badge badge bg-danger';
                }
                
                // Set last updated time
                document.getElementById('lastUpdatedTime').textContent = new Date().toLocaleTimeString();
                
                // Add pulse animation to show data has been updated
                document.querySelectorAll('.health-item').forEach(item => {
                    item.classList.add('pulse-animation');
                    setTimeout(() => {
                        item.classList.remove('pulse-animation');
                    }, 500);
                });
            })
            .catch(error => {
                console.error('Error updating health dashboard:', error);
                document.getElementById('lastUpdatedTime').textContent = 'Failed to update';
            });
    }
});

// Function to toggle citations visibility - make it global so it can be called from HTML
window.toggleCitations = function(element) {
    console.log("Toggle citations called", element);
    const container = element.closest('.citations-container');
    const content = container.querySelector('.citations-content');
    const icon = element.querySelector('.bi-chevron-down, .bi-chevron-up');
    
    console.log("Citations container:", container);
    console.log("Citations content:", content);
    console.log("Citations icon:", icon);
    
    content.classList.toggle('open');
    console.log("Toggled 'open' class:", content.classList.contains('open'));
    
    if (content.classList.contains('open')) {
        icon.classList.remove('bi-chevron-down');
        icon.classList.add('bi-chevron-up');
    } else {
        icon.classList.remove('bi-chevron-up');
        icon.classList.add('bi-chevron-down');
    }
};
