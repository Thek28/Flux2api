from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import json
import time
import os
import re  # Import the 're' module
import math
import random
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("警告：zhipuai 库未安装，智谱文生图功能将不可用。请使用 pip install zhipuai 安装。")
    ZhipuAI = None

app = Flask(__name__)

# 代理配置
HTTP_PROXY = os.environ.get("HTTP_PROXY", "")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY", "")
API_KEY = os.environ.get("API_KEY", "")
FAL_API_KEY = os.environ.get("FAL_API_KEY", "8aca507c-281a-4a27-a716-853bbab6ed71:3f0f85fb85d4f0129c84ef61d060dbe6")
GML_API_KEY = os.environ.get("GML_API_KEY", "0e78255708974e158a8369212f0092b9.YojuGtv40BRAik8S")  # 从环境变量获取智谱API密钥

FAL_API_KEY_LIST = FAL_API_KEY.split(",") if FAL_API_KEY else []
GML_API_KEY_LIST = GML_API_KEY.split(",") if GML_API_KEY else []  # 支持多个API KEY轮询

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
    },
    "cogview-4-250304": {
        "provider": "zhipuai",  # 标记为智谱提供商
        "model": "cogview-4-250304"  # 智谱模型名称
    }
}

def make_request(api_key: str, prompt: str):
    url = "https://api.oaiopen.cn/v1/chat/completions"
    models = ["grok-3-beta-flux"]
    model = random.choice(models)
    prompt_system = f"""您是一位才华横溢的 AI 画家，擅 长创作富有想象力和视觉冲击力的数字艺术作品。您的任务是根据用户的描述，创作出令人惊 叹的图像。请遵循以下指南：

    1. 仔细分析用户的需求，确保理解 所有关键元素。如有不清楚的地方，请礼貌地询问用户以获取更多细节。

    2 . 在创作过程中，请考虑以下要素：
       - 主体：明确描述画面中的主要对象（ 如人物、动物、建筑或物体）
       - 媒介：指定作品的艺术形式（如 照片、油画、水彩、插画或数字艺术）
       - 环境：详细描述背景场景（ 如自然风光、城市街道或抽象空间）
       - 光线：说明画面的光源和 光线效果（如自然光、人工光或特殊光效）
       - 颜色：指定作品的色 彩方案（如明亮、柔和、单色或多彩）
       - 情绪：传达作品应表现的情感氛围 （如欢快、忧郁、神秘或激情）
       - 构图/角度：描述 画面的构图和视角（如特写、全景、俯视或仰角）

    3 . 根据用户的描述，综合运用这些元素创作出独特而吸引人的图像描述。

     4. 如果用户要求修改或提供反馈，请认真聆听并相 应调整您的创作。

    5. 始终保持友好、专业的态度，展现出您作为 AI 画家的创造力和 艺术素养。

    ## 重要
    1. 整体内容在50-150字。
    2. 请概 括成一段话输出。
    3. 必须用英文输出。
    4. 请严格遵循下面回复示 例格式。

    ## 示例

    用户发送：画一个二战时期的护士 

    您回复：

    ```prompt
    A WWII-era nurse in a German uniform, holding a wine bottle and stethoscope , sitting at a table in white attire, with a table in the background, masterpiece, best quality, 4k , illustration style, best lighting, depth of field, detailed character, detailed environment.
    ```
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ]
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

def get_gml_api_key():
    # 随机获取 GML_API_KEY_LIST
    if GML_API_KEY_LIST:
        return random.choice(GML_API_KEY_LIST)
    raise ValueError("GML_API_KEY is not set.")

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

def call_gml_api(prompt, model, options=None):
    """
    调用智谱API生成图像
    
    Args:
        prompt: 用于生成图像的文本提示
        model: 使用的模型名称
        options: 附加选项，如num_images等
        
    Returns:
        成功时返回图像URL列表，失败时抛出异常
    """
    if ZhipuAI is None:
        raise ValueError("智谱AI SDK未安装，请使用pip install zhipuai安装")
    
    if options is None:
        options = {}
    
    # 准备基本请求参数
    num_images = options.get("num_images", 1)
    max_retries = 3
    retry_count = 0
    image_urls = []
    
    while retry_count <= max_retries:
        try:
            # 获取API密钥
            gml_api_key = get_gml_api_key()
            print(f"Attempt {retry_count + 1}/{max_retries + 1} - Using ZhipuAI key: {gml_api_key[:5]}...{gml_api_key[-5:] if len(gml_api_key) > 10 else ''}")
            
            # 调用智谱API
            client = ZhipuAI(api_key=gml_api_key)
            response = client.images.generations(
                model=model,
                prompt=prompt,
                n=num_images  # 生成图片数量
            )
            
            # 处理响应
            if hasattr(response, 'data') and response.data:
                for item in response.data:
                    if hasattr(item, 'url') and item.url:
                        image_urls.append(item.url)
                        print(f"Found ZhipuAI image URL: {item.url}")
            
            if image_urls:
                break
                
        except Exception as e:
            error_message = str(e)
            print(f"ZhipuAI API error: {error_message}")
            
            # 处理认证错误
            if "api_key" in error_message.lower() or "unauthorized" in error_message.lower():
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"ZhipuAI API key认证失败，尝试使用新的API key重试 ({retry_count}/{max_retries})")
                    time.sleep(2 ** retry_count)
                    continue
                else:
                    raise ValueError(f"Authentication error with ZhipuAI API after {max_retries} retries: {error_message}")
            
            # 处理其他错误
            if retry_count < max_retries:
                retry_count += 1
                print(f"ZhipuAI API错误，进行重试 ({retry_count}/{max_retries}): {error_message}")
                time.sleep(2 ** retry_count)
                continue
            
            raise ValueError(f"Error calling ZhipuAI API: {error_message}")
    
    if not image_urls:
        raise ValueError("No images found from ZhipuAI API after multiple attempts")
    
    return image_urls

def call_model_api(prompt, model, options=None):
    """
    统一的API调用入口，根据模型类型分发到不同的API调用函数
    
    Args:
        prompt: 用于生成图像的文本提示
        model: 使用的模型名称
        options: 附加选项
        
    Returns:
        成功时返回图像URL列表，失败时抛出异常
    """
    # 获取模型信息
    model_info = MODEL_URLS.get(model)
    if not model_info:
        raise ValueError(f"Unsupported model: {model}")
    
    # 根据提供商分发到不同的API调用函数
    provider = model_info.get("provider")
    if provider == "zhipuai":
        return call_gml_api(prompt, model_info.get("model"), options)
    else:
        return call_fal_api(prompt, model, options)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    处理聊天完成请求，路由到不同的模型。
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
    
    # 针对智谱AI模型处理 - 直接使用原始提示词而不做转换
    if model == "cogview-4-250304":
        print("『执行』: 智谱模型使用原始提示词")
    else:
        # 非智谱模型，使用转换后的提示词
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
        # 调用API生成图像 - 使用统一入口
        image_urls = call_model_api(prompt, model)

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
    num_images = openai_request.get('n', 1)  # 添加支持生成多张图片
    
    # 打印原始提示词
    print("『执行』: 图像生成原始提示词：" + prompt)
    
    # 针对非智谱AI模型，转换提示词
    if model != "cogview-4-250304":
        try:
            # 尝试转换提示词
            converted_prompt = make_request('sk-OUISlfp3DZsJNRaV89676536131e43A88fBd61A80b7739C6', prompt)
            if converted_prompt:
                prompt = converted_prompt
                print("『执行』: 图像生成转换后提示词：" + prompt)
        except Exception as e:
            print(f"『执行』: 提示词转换失败: {str(e)}")
    else:
        print("『执行』: 智谱模型使用原始提示词")

    # 准备选项参数
    options = {
        "size": size,
        "seed": seed,
        "output_format": output_format,
        "num_images": num_images
    }

    try:
        # 使用统一的API调用入口
        image_urls = call_model_api(prompt, model, options)

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
        if "Authentication error" in error_message or "api_key" in error_message.lower() or "unauthorized" in error_message.lower():
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
                "type": "api_error",
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
         "owned_by": "fal-openai-adapter", "permission": [], "root": "ideogram-v2", "parent": None},
        {"id": "cogview-4-250304", "object": "model", "created": 1698785189,
         "owned_by": "zhipuai-adapter", "permission": [], "root": "cogview-4-250304", "parent": None}
    ]
    return jsonify({"object": "list", "data": models})


if __name__ == "__main__":
    # 检查环境变量
    missing_keys = []
    if not FAL_API_KEY:
        print("警告: FAL_API_KEY 未设置。部分功能可能不可用。")
        missing_keys.append("FAL_API_KEY")
    
    if not GML_API_KEY and ZhipuAI is not None:
        print("警告: GML_API_KEY 未设置。智谱文生图功能将不可用。")
        missing_keys.append("GML_API_KEY")
    
    if missing_keys:
        print(f"缺少以下环境变量: {', '.join(missing_keys)}")
        if "FAL_API_KEY" in missing_keys:
            raise ValueError("FAL_API_KEY is not set.")

    port = int(os.environ.get("PORT", 5005))
    print(f"服务启动于端口 {port}...")
    if HTTP_PROXY or HTTPS_PROXY:
        print(f"HTTP代理: {HTTP_PROXY}")
        print(f"HTTPS代理: {HTTPS_PROXY}")
    
    # 打印可用的图像生成模型
    print("可用的图像生成模型:")
    for model_name, model_info in MODEL_URLS.items():
        provider = model_info.get("provider", "fal-ai")
        print(f"  - {model_name} (提供商: {provider})")
    
    app.run(host='0.0.0.0', port=port, debug=True)
