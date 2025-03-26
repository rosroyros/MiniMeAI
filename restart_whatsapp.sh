#!/bin/bash
# Script to restart the WhatsApp bridge with the new pull-based architecture

echo "Restarting WhatsApp bridge with new pull-based architecture..."

# SSH to the Raspberry Pi
ssh roy@10.0.0.58 "cd /home/roy/MiniMeAI && \
  echo 'Stopping and removing WhatsApp bridge...' && \
  docker-compose stop whatsapp_bridge && \
  docker-compose rm -f whatsapp_bridge && \
  echo 'Rebuilding WhatsApp bridge...' && \
  docker-compose build whatsapp_bridge && \
  echo 'Starting WhatsApp bridge with new configuration...' && \
  docker-compose up -d whatsapp_bridge && \
  echo 'WhatsApp bridge restarted successfully.'"

echo "To view logs, run: ssh roy@10.0.0.58 \"cd /home/roy/MiniMeAI && docker-compose logs --follow whatsapp_bridge\""
echo "Done!" 