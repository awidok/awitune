#!/usr/bin/env python3
"""
OpenAI to Anthropic API Proxy Server

Converts OpenAI-compatible API requests to Anthropic format using Flask.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import urllib3
from dotenv import load_dotenv

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Configure logging — log to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/proxy_debug.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'http://localhost:11434/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
PROXY_PORT = int(os.getenv('PROXY_PORT', 8080))
PROXY_HOST = os.getenv('PROXY_HOST', 'localhost')

# Create Flask app
app = Flask(__name__)
CORS(app)


def convert_anthropic_to_openai(messages):
    """Convert Anthropic message format to OpenAI format."""
    openai_messages = []

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Handle complex content (arrays with text/tool_use/tool_result parts)
        if isinstance(content, list):
            # Extract text parts
            text_parts = [
                part.get('text', '')
                for part in content
                if part.get('type') == 'text'
            ]

            # Extract tool_use parts and convert to OpenAI tool_calls format
            tool_uses = [
                part for part in content
                if part.get('type') == 'tool_use'
            ]

            # Extract tool_result parts and convert to separate tool messages
            tool_results = [
                part for part in content
                if part.get('type') == 'tool_result'
            ]

            # Add main message with text content
            if text_parts or tool_uses:
                new_msg = {
                    'role': role,
                    'content': '\n'.join(text_parts) if text_parts else ''
                }

                # Add tool_calls if present
                if tool_uses:
                    new_msg['tool_calls'] = [
                        {
                            'id': tool.get('id'),
                            'type': 'function',
                            'function': {
                                'name': tool.get('name'),
                                'arguments': json.dumps(tool.get('input', {}))
                            }
                        }
                        for tool in tool_uses
                    ]

                openai_messages.append(new_msg)

            # Add tool result messages
            for tool_result in tool_results:
                tr_content = tool_result.get('content', '')
                # content can be a string or a list of content blocks
                if isinstance(tr_content, list):
                    tr_text = '\n'.join(
                        part.get('text', '') for part in tr_content
                        if isinstance(part, dict) and part.get('type') == 'text'
                    )
                elif isinstance(tr_content, str):
                    tr_text = tr_content
                else:
                    tr_text = str(tr_content) if tr_content else ''
                openai_messages.append({
                    'role': 'tool',
                    'content': tr_text or tool_result.get('text', ''),
                    'tool_call_id': tool_result.get('tool_use_id')
                })
        else:
            openai_messages.append({
                'role': role,
                'content': content
            })

    return openai_messages


def convert_openai_to_anthropic(openai_response, model):
    """Convert OpenAI response format to Anthropic format."""
    choice = openai_response.get('choices', [{}])[0]
    message = choice.get('message', {})

    # Build content blocks array
    content_blocks = []

    # Add text content if present
    # Some models (e.g., GLM-5) put content in 'reasoning_content' instead of 'content'
    text_content = message.get('content', '') or ''
    reasoning_content = message.get('reasoning_content', '') or ''

    # Combine: prefer content, fall back to reasoning_content
    final_text = text_content or reasoning_content
    if final_text:
        content_blocks.append({
            'type': 'text',
            'text': final_text
        })

    # Convert tool_calls to tool_use blocks
    tool_calls = message.get('tool_calls', []) or []
    for tool_call in tool_calls:
        if tool_call.get('type') == 'function':
            func = tool_call.get('function', {})
            try:
                arguments = json.loads(func.get('arguments', '{}'))
            except (json.JSONDecodeError, ValueError, TypeError):
                arguments = {}

            content_blocks.append({
                'type': 'tool_use',
                'id': tool_call.get('id', f"toolu_{int(datetime.now().timestamp())}"),
                'name': func.get('name', ''),
                'input': arguments
            })

    # Map finish_reason to stop_reason
    finish_reason = choice.get('finish_reason', 'stop')
    if finish_reason == 'tool_calls':
        stop_reason = 'tool_use'
    elif finish_reason == 'length':
        stop_reason = 'max_tokens'
    else:
        stop_reason = 'end_turn'

    return {
        'id': openai_response.get('id', f"msg_{int(datetime.now().timestamp())}").replace('chatcmpl', 'msg'),
        'type': 'message',
        'role': 'assistant',
        'content': content_blocks,
        'model': 'claude-opus-4-6',
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {
            'input_tokens': openai_response.get('usage', {}).get('prompt_tokens', 0),
            'output_tokens': openai_response.get('usage', {}).get('completion_tokens', 0)
        }
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'openai-to-anthropic-proxy',
        'version': '1.0.0'
    })


@app.route('/v1/messages', methods=['POST'])
def create_message():
    """Main endpoint for Anthropic messages API."""
    try:
        data = request.get_json()

        model = data.get('model', 'default')
        messages = data.get('messages', []) or []
        max_tokens = data.get('max_tokens', 16000)
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', float(os.getenv('TOP_P', 0.95)))
        system = data.get('system')
        tools = data.get('tools', []) or []
        stream = data.get('stream', False)

        logger.info(f"Processing request for model: {model}")

        # Convert messages
        openai_messages = convert_anthropic_to_openai(messages)

        # Add system message if present
        if system:
            # System can be a string or array of content blocks
            if isinstance(system, list):
                system_text = '\n'.join([
                    item.get('text', '') for item in system
                    if item.get('type') == 'text'
                ])
            else:
                system_text = system
            openai_messages.insert(0, {'role': 'system', 'content': system_text})

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            if tool.get('name') in ['BatchTool']:  # Skip certain tools
                continue
            openai_tools.append({
                'type': 'function',
                'function': {
                    'name': tool.get('name'),
                    'description': tool.get('description', ''),
                    'parameters': tool.get('input_schema', {})
                }
            })

        # Build OpenAI request - always use "default" model for GLM-5
        openai_request = {
            'model': 'default',
            'messages': openai_messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }

        if openai_tools:
            openai_request['tools'] = openai_tools

        # Make request to OpenAI-compatible API
        print(f"\n=== Sending to GLM-5 ===")
        print(f"Model: {openai_request['model']}")
        print(f"Messages count: {len(openai_request['messages'])}")
        print(f"Max tokens: {openai_request['max_tokens']}")
        if openai_tools:
            print(f"Tools: {len(openai_tools)}")

        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json=openai_request,
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            timeout=300,  # 5 minutes
            verify=False,  # Allow self-signed certificates
        )

        print(f"GLM-5 response status: {response.status_code}")
        print(f"GLM-5 response length: {len(response.text)}")
        print(f"GLM-5 response preview: {response.text[:500]}")
        response.raise_for_status()

        # Convert response to Anthropic format
        try:
            openai_data = response.json()
        except ValueError as json_err:
            logger.error(f"JSON decode error. Response status: {response.status_code}, Content: {response.text}")
            return jsonify({
                'type': 'error',
                'error': {
                    'type': 'api_error',
                    'message': f'Invalid JSON response from upstream API: {str(json_err)}'
                }
            }), 500

        print(f"OpenAI response data: {json.dumps(openai_data, indent=2)[:1000]}")
        anthropic_response = convert_openai_to_anthropic(openai_data, model)
        print(f"Anthropic response: {json.dumps(anthropic_response, indent=2)[:1000]}")
        logger.debug(f"Returning to Claude Code: {str(anthropic_response)[:1000]}")

        return jsonify(anthropic_response)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        # Try to get response details if available
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response headers: {dict(e.response.headers)}")
            logger.error(f"Response body: {e.response.text[:2000]}")
        return jsonify({
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': str(e)
            }
        }), 500

    except Exception as e:
        import traceback
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': str(e)
            }
        }), 500


@app.route('/v1/messages/count_tokens', methods=['POST'])
def count_tokens():
    """Count tokens endpoint - returns estimated token count."""
    try:
        data = request.get_json()

        # Simple estimation: count words/characters
        # This is a rough approximation since we don't have access to the actual tokenizer
        messages = data.get('messages', []) or []
        system = data.get('system', '')

        total_chars = 0

        # Count system message
        if system:
            if isinstance(system, list):
                for item in system:
                    if item.get('type') == 'text':
                        total_chars += len(item.get('text', ''))
            else:
                total_chars += len(system)

        # Count messages
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        total_chars += len(item.get('text', ''))

        # Rough estimate: ~4 chars per token
        estimated_tokens = max(1, total_chars // 4)

        return jsonify({
            'input_tokens': estimated_tokens
        })

    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return jsonify({
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': str(e)
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models."""
    try:
        response = requests.get(
            f"{OPENAI_API_BASE}/models",
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            timeout=30,
            verify=False,
        )

        response.raise_for_status()
        openai_models = response.json()

        # Convert to Anthropic format
        models = [
            {
                'id': model['id'],
                'type': 'model',
                'display_name': model['id'],
                'created_at': datetime.now().isoformat()
            }
            for model in openai_models.get('data', [])
        ]

        return jsonify({'data': models})

    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return jsonify({
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': str(e)
            }
        }), 500


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def catch_all(path):
    """Catch-all route for unhandled endpoints — log and return OK."""
    logger.warning(f"Unhandled request: {request.method} /{path}")
    logger.warning(f"  Headers: {dict(request.headers)}")
    body = request.get_data(as_text=True)
    if body:
        logger.warning(f"  Body: {body[:500]}")

    # Return a generic success response
    return jsonify({
        'status': 'ok',
        'message': f'Unhandled endpoint: /{path}'
    }), 200


def main():
    """Main entry point."""
    # Enable Flask request logging
    import logging as log
    log.getLogger('werkzeug').setLevel(log.INFO)

    print("\n🚀 OpenAI to Anthropic API Proxy Server started")
    print(f"   URL: http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"   Target: {OPENAI_API_BASE}")
    print("\nConfigure Claude Code with:")
    print(f'   export ANTHROPIC_API_KEY="<your-api-key>"')
    print(f'   export ANTHROPIC_BASE_URL="http://{PROXY_HOST}:{PROXY_PORT}"\n')

    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()
