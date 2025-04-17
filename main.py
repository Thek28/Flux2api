from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import json
import time
import os
import re  # Import the 're' module
import math
import random

app = Flask(__name__)

# 代理配置
HTTP_PROXY = os.environ.get("HTTP_PROXY", "")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY", "")
API_KEY = os.environ.get("API_KEY", "")
FAL_API_KEY = os.environ.get("FAL_API_KEY", "8aca507c-281a-4a27-a716-853bbab6ed71:3f0f85fb85d4f0129c84ef61d060dbe6")

FAL_API_KEY_LIST = FAL_API_KEY.split(",") if FAL_API_KEY else []

# Restructured MODEL_URLS to separate submit and status/result URLs
MODEL_URLS = {
    "flux-1.1-ultra": {
        "submit_url": "https://queue.fal.run/fal-ai/flux-pro/v1.1-ultra",
        "status_base_url": "https://queue.fal.run/fal-ai/flux-pro"
    },
    "recraft-v3": {
        "submit_url": "https://queue.fal.run/fal-ai/recraft-v3",
        "status_base_url": "https://queue.fal.run/fal-ai/recraft-v3"
    },
    "flux-1.1-pro": {
        "submit_url": "https://queue.fal.run/fal-ai/flux-pro/v1.1",
        "status_base_url": "https://queue.fal.run/fal-ai/flux-pro"
    },
    "ideogram-v2": {
        "submit_url": "https://queue.fal.run/fal-ai/ideogram/v2",
        "status_base_url": "https://queue.fal.run/fal-ai/ideogram"
    },
    "flux-dev": {
        "submit_url": "https://queue.fal.run/fal-ai/flux/dev",
        "status_base_url": "https://queue.fal.run/fal-ai/flux"
    }
}

def make_request(api_key: str, prompt: str):
    url = "https://api.oaiopen.cn/v1/chat/completions"
    models = ["grok-3-beta"]
    model = random.choice(models)
    prompt = "你现在是一个高级提示词专家，你只需要把翻译后的结果发给我不需要其他的回复性措辞，请帮我把下面这一段中文提示词转成Flux图形模型接口能识别的英文版，需要全英文且标签化的提示词：" + prompt
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_text = response.text
        parsed_json_data = json.loads(response_text)
        # 提取 content 的值
        content = parsed_json_data['choices'][0]['message']['content']
        print(f"『执行』: 获取到Flux专业的提示词：{content}")
        return content
    except json.JSONDecodeError:
        print(f"『执行』: 获取到Flux专业的提示词-从响应解码JSON时出错: {response_text}")
        return None
    except Exception as e:
        print(f"『执行』: 获取到Flux专业的提示词-请求失败: {e}")
        return None

def get_fal_api_key():
    # 随机获取 FAL_API_KEY_LIST
    if FAL_API_KEY_LIST:
        return random.choice(FAL_API_KEY_LIST)
    raise ValueError("FAL_API_KEY is not set.")


def validate_api_key(api_key):
    if API_KEY and API_KEY != api_key:
        return False
    return True


def get_proxies():
    """获取代理配置"""

    proxies = {}
    if HTTP_PROXY:
        proxies["http"] = HTTP_PROXY
    if HTTPS_PROXY:
        proxies["https"] = HTTPS_PROXY

    print(f"Using proxies: {proxies}")
    return proxies if proxies else None


def call_fal_api(prompt, model, options=None):
    """
    通用方法用于调用Fal API生成图像

    Args:
        prompt: 用于生成图像的文本提示
        model: 使用的模型名称
        options: 附加选项，如size、seed等

    Returns:
        成功时返回图像URL列表，失败时抛出异常
    """
    if options is None:
        options = {}

    # 准备基本请求参数
    fal_request = {
        "prompt": prompt,
        "num_images": options.get("num_images", 1)
    }

    # 添加其他可选参数
    if "seed" in options:
        fal_request["seed"] = options["seed"]
    if "output_format" in options:
        fal_request["output_format"] = options["output_format"]

    # 处理图像尺寸或宽高比
    if "size" in options:
        width, height = map(int, options["size"].split("x"))
        if model == "flux-1.1-ultra" or model == "ideogram-v2":
            gcd = math.gcd(width, height)
            fal_request["aspect_ratio"] = f"{width // gcd}:{height // gcd}"
        else:
            fal_request["image_size"] = {"width": width, "height": height}

    # 获取模型URL信息
    fal_submit_url = MODEL_URLS.get(model, MODEL_URLS["flux-dev"])["submit_url"]
    fal_status_base_url = MODEL_URLS.get(model, MODEL_URLS["flux-dev"])["status_base_url"]
    print(f"Using model: {model}, Submit URL: {fal_submit_url}")
    print(f"Request data: {json.dumps(fal_request)}")

    # 添加重试逻辑
    max_retries = 3
    retry_count = 0
    image_urls = []

    while retry_count <= max_retries:
        try:
            # 获取API密钥和代理设置
            fal_api_key = get_fal_api_key()
            headers = {
                "Authorization": f"Key {fal_api_key}",
                "Content-Type": "application/json"
            }
            proxies = get_proxies()

            print(
                f"Attempt {retry_count + 1}/{max_retries + 1} - Using key: {fal_api_key[:5]}...{fal_api_key[-5:] if len(fal_api_key) > 10 else ''}")

            # 提交请求
            session = requests.Session()
            fal_response = session.post(
                fal_submit_url,
                headers=headers,
                json=fal_request,
                proxies=proxies,
                timeout=30
            )

            if fal_response.status_code != 200:
                # 处理错误响应
                try:
                    error_data = fal_response.json()
                    error_message = error_data.get('error', {}).get('message', fal_response.text)
                except:
                    error_message = fal_response.text

                print(f"Fal API error: {fal_response.status_code}, {error_message}")

                # 处理认证错误
                if fal_response.status_code in (401, 403):
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"API key认证失败，尝试使用新的API key重试 ({retry_count}/{max_retries})")
                        time.sleep(2 ** retry_count)
                        continue
                    else:
                        raise ValueError(
                            f"Authentication error with Fal API after {max_retries} retries: {error_message}")

                # 处理其他错误
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Fal API错误，进行重试 ({retry_count}/{max_retries})")
                    time.sleep(2 ** retry_count)
                    continue

                raise ValueError(f"Fal API error after {max_retries} retries: {error_message}")

            # 解析响应获取请求ID
            fal_data = fal_response.json()
            request_id = fal_data.get("request_id")
            if not request_id:
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"未获取request_id，进行重试 ({retry_count}/{max_retries})")
                    time.sleep(2 ** retry_count)
                    continue
                raise ValueError("Missing request_id in Fal API response")

            print(f"Got request_id: {request_id}")

            # 轮询获取结果
            max_polling_attempts = 60
            for attempt in range(max_polling_attempts):
                print(f"Polling attempt {attempt + 1}/{max_polling_attempts}")
                try:
                    # 构建状态和结果URL
                    status_url = f"{fal_status_base_url}/requests/{request_id}/status"
                    result_url = f"{fal_status_base_url}/requests/{request_id}"

                    # 检查状态
                    status_session = requests.Session()
                    status_response = status_session.get(
                        status_url,
                        headers=headers,
                        proxies=proxies,
                        timeout=30
                    )

                    # 处理认证失败
                    if status_response.status_code in (401, 403):
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"状态检查认证失败，尝试使用新的API key重试 ({retry_count}/{max_retries})")
                            time.sleep(2 ** retry_count)
                            break  # 跳出状态检查循环，重新开始外部尝试
                        else:
                            raise ValueError(f"Authentication error during status check after {max_retries} retries")

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")

                        # 处理失败状态
                        if status == "FAILED":
                            if retry_count < max_retries:
                                retry_count += 1
                                print(f"生成失败，进行重试 ({retry_count}/{max_retries})")
                                time.sleep(2 ** retry_count)
                                break  # 跳出状态检查循环，重新开始外部尝试
                            raise ValueError("Image generation failed")

                        # 处理完成状态
                        if status == "COMPLETED":
                            print(f"Fetching result from: {result_url}")

                            # 获取结果
                            result_session = requests.Session()
                            result_response = result_session.get(
                                result_url,
                                headers={"Authorization": f"Key {fal_api_key}"},
                                proxies=proxies,
                                timeout=30
                            )

                            # 处理结果获取的认证失败
                            if result_response.status_code in (401, 403):
                                if retry_count < max_retries:
                                    retry_count += 1
                                    print(f"结果获取认证失败，尝试使用新的API key重试 ({retry_count}/{max_retries})")
                                    time.sleep(2 ** retry_count)
                                    break  # 跳出状态检查循环，重新开始外部尝试
                                else:
                                    raise ValueError(
                                        f"Authentication error during result fetch after {max_retries} retries")

                            if result_response.status_code == 200:
                                result_data = result_response.json()

                                # 提取图片URL
                                if "images" in result_data:
                                    images = result_data.get("images", [])
                                    for img in images:
                                        if isinstance(img, dict) and "url" in img:
                                            image_urls.append(img.get("url"))
                                            print(f"Found image URL: {img.get('url')}")

                                if image_urls:
                                    break
                                elif retry_count < max_retries:
                                    retry_count += 1
                                    print(f"未找到图片，进行重试 ({retry_count}/{max_retries})")
                                    time.sleep(2 ** retry_count)
                                    break  # 跳出状态检查循环，重新开始外部尝试

                        time.sleep(2)
                    else:
                        # print(f"Error checking status: {status_response.text}")
                        time.sleep(2)

                except Exception as e:
                    print(f"Error during polling: {str(e)}")
                    time.sleep(2)

                # 如果已经获取到图片URLs，退出状态检查循环
                if image_urls:
                    break

            # 如果已经获取到图片URLs，退出整个重试循环
            if image_urls:
                break

            # 如果状态检查完成但没有获取到图片，并且还有重试次数，则继续重试
            if not image_urls and retry_count < max_retries:
                retry_count += 1
                print(f"未获取到图片URL，进行整体重试 ({retry_count}/{max_retries})")
                time.sleep(2 ** retry_count)
                continue

            # 如果已经没有重试次数且没有图片，则抛出异常
            if not image_urls:
                raise ValueError("No images found after multiple attempts")

        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                print(f"发生异常，进行重试 ({retry_count}/{max_retries}): {str(e)}")
                time.sleep(2 ** retry_count)
                continue
            raise ValueError(f"Error calling Fal API: {str(e)}")

    return image_urls


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    处理聊天完成请求，路由到不同的Fal模型。
    支持流式输出和标准响应格式。
    """
    auth_header = request.headers.get('Authorization', '')
    print(f"Received Authorization header: {auth_header}")

    if auth_header.startswith('Bearer '):
        api_key = auth_header[7:]
    elif auth_header.startswith('Key '):
        api_key = auth_header[4:]
    else:
        api_key = auth_header

    if not validate_api_key(api_key):
        print("Invalid API key provided")
        return jsonify({
            "error": {
                "message": "Invalid API key provided",
                "type": "invalid_api_key"
            }
        }), 401

    openai_request = request.json
    if not openai_request:
        return jsonify({
            "error": {
                "message": "Missing or invalid request body",
                "type": "invalid_request_error"
            }
        }), 400

    # 检查是否请求流式输出
    stream = openai_request.get('stream', False)
    messages = openai_request.get('messages', [])
    model = openai_request.get('model', 'flux-1.1-ultra')  # Default

    # 使用最后一条用户消息作为提示
    prompt = ""
    last_user_message = next((msg['content'] for msg in reversed(messages) if msg.get('role') == 'user'), None)
    if last_user_message:
        prompt = last_user_message
    print("『执行』: 用户发送的【原】提示词为：" + prompt)
    prompt = make_request('sk-OUISlfp3DZsJNRaV89676536131e43A88fBd61A80b7739C6', prompt)
    print("『执行』: 用户发送的【新】提示词为：" + prompt)
    if not prompt:
        # 如果没有提示词，返回默认响应
        if stream:
            def generate():
                current_time = int(time.time())
                # 首先发送正在思考的消息
                response_start = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_start)}\n\n"

                # 分段发送内容
                response_content = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "I can generate images. Describe what you'd like."
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_content)}\n\n"

                # 最后发送完成标记
                response_end = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(response_end)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            # 非流式响应
            completions_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I can generate images. Describe what you'd like."
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(json.dumps(messages)) // 4,
                    "completion_tokens": 20,
                    "total_tokens": (len(json.dumps(messages)) // 4) + 20
                },
                "system_fingerprint": f"fp_{int(time.time())}"
            }
            return jsonify(completions_response)

    try:
        # 调用Fal API
        image_urls = call_fal_api(prompt, model)

        # 构建响应内容
        content = ""
        for i, url in enumerate(image_urls):
            if i > 0:
                content += "\n\n"
            content += f"![Generated Image {i + 1}]({url}) "

        if stream:
            # 流式输出
            def generate():
                current_time = int(time.time())

                # 首先发送角色
                response_role = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_role)}\n\n"

                # 然后直接发送图片链接(作为Markdown格式)
                response_content = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_content)}\n\n"

                # 最后发送完成标记
                response_end = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(response_end)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            # 生成标准OpenAI格式的响应
            completions_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(prompt) // 4) + (len(content) // 4)
                },
                "system_fingerprint": f"fp_{int(time.time())}"
            }

            print(f"Returning OpenAI completions-style response")
            return jsonify(completions_response)

    except ValueError as e:
        # 处理已知错误
        print(f"Error: {str(e)}")
        error_message = f"Unable to generate an image: {str(e)}. Try a different description."

        if stream:
            # 流式输出错误信息
            def generate():
                current_time = int(time.time())

                # 发送角色
                response_role = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_role)}\n\n"

                # 发送错误信息
                response_error = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": error_message
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_error)}\n\n"

                # 发送完成标记
                response_end = {
                    "id": f"chatcmpl-{current_time}",
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(response_end)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            # 返回标准错误响应
            completions_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": error_message
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(error_message) // 4,
                    "total_tokens": (len(prompt) // 4) + (len(error_message) // 4)
                },
                "system_fingerprint": f"fp_{int(time.time())}"
            }
            return jsonify(completions_response)

    except Exception as e:
        # 处理未知错误
        print(f"Exception: {str(e)}")
        return jsonify({"error": {"message": f"Server error: {str(e)}", "type": "server_error"}}), 500


@app.route('/v1/images/generations', methods=['POST'])
def generate_image():
    """Legacy endpoint for direct image generations."""
    auth_header = request.headers.get('Authorization', '')

    if auth_header.startswith('Bearer '):
        api_key = auth_header[7:]
    elif auth_header.startswith('Key '):
        api_key = auth_header[4:]
    else:
        api_key = auth_header

    if not validate_api_key(api_key):
        print("Invalid API key provided")
        return jsonify({
            "error": {
                "message": "Invalid API key provided",
                "type": "invalid_api_key"
            }
        }), 401

    openai_request = request.json
    if not openai_request:
        return jsonify({"error": {"message": "Missing or invalid request body", "type": "invalid_request_error"}}), 400

    prompt = openai_request.get('prompt', '')
    if not prompt:
        return jsonify({
            "error": {
                "message": "prompt is required",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400

    # 提取请求参数
    model = openai_request.get('model', 'flux-dev')
    size = openai_request.get('size', '1080x1920')
    seed = openai_request.get('seed', openai_request.get('user', 100010))
    output_format = openai_request.get('response_format', 'jpeg')

    # 准备选项参数
    options = {
        "size": size,
        "seed": seed,
        "output_format": output_format,
        "num_images": 1
    }

    try:
        # 调用Fal API
        image_urls = call_fal_api(prompt, model, options)

        # 构建OpenAI格式的响应
        data = [{"url": url} for url in image_urls]
        completions_response = {
            "created": int(time.time()),
            "model": model,
            "data": data,
        }
        return jsonify(completions_response)

    except ValueError as e:
        # 处理已知错误
        error_message = str(e)

        # 检查是否是认证错误
        if "Authentication error" in error_message:
            return jsonify({
                "error": {
                    "message": error_message,
                    "type": "invalid_api_key",
                    "code": 401
                }
            }), 401

        # 其他API错误
        return jsonify({
            "error": {
                "message": error_message,
                "type": "fal_api_error",
                "code": 500
            }
        }), 500

    except Exception as e:
        # 处理未知错误
        print(f"Exception: {str(e)}")
        return jsonify({"error": {"message": f"Server error: {str(e)}", "type": "server_error"}}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """Mock OpenAI models endpoint."""
    models = [
        {"id": "flux-dev", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "flux-dev", "parent": None},
        {"id": "flux-1.1-ultra", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "flux-1.1-ultra", "parent": None},
        {"id": "recraft-v3", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "recraft-v3", "parent": None},
        {"id": "flux-1.1-pro", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "flux-1.1-pro", "parent": None},
        {"id": "ideogram-v2", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "ideogram-v2", "parent": None}
    ]
    return jsonify({"object": "list", "data": models})


if __name__ == "__main__":
    if not FAL_API_KEY:
        print("Warning: FAL_API_KEY is not set. Some features may not work.")
        raise ValueError("FAL_API_KEY is not set.")

    port = int(os.environ.get("PORT", 5005))
    print(f"Starting server on port {port}...")
    if HTTP_PROXY or HTTPS_PROXY:
        print(f"HTTP Proxy: {HTTP_PROXY}")
        print(f"HTTPS Proxy: {HTTPS_PROXY}")
    app.run(host='0.0.0.0', port=port, debug=True)