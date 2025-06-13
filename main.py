from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import json
import time
import os
import re  # Import the 're' module
import math
import random
import jwt  # 添加jwt导入，用于可灵AI认证
import urllib3  # 添加urllib3导入
import base64  # 添加base64导入，用于图片编码
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("警告：zhipuai 库未安装，智谱文生图功能将不可用。请使用 pip install zhipuai 安装。")
    ZhipuAI = None

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# 代理配置
HTTP_PROXY = os.environ.get("HTTP_PROXY", "")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY", "")
API_KEY = os.environ.get("API_KEY", "")
FAL_API_KEY = os.environ.get("FAL_API_KEY", "f85686c1-75f9-40e5-93fd-ef84be83d4e9:97fa06aeaecf1fb2d873a0a78a90c7d7")
GML_API_KEY = os.environ.get("GML_API_KEY", "0e78255708974e158a8369212f0092b9.YojuGtv40BRAik8S")  # 从环境变量获取智谱API密钥
KLING_ACCESS_KEY = os.environ.get("KLING_ACCESS_KEY", "Abnhn3hn3HTY4bDpRCPRtTygQKnagLYf")  # 可灵AI Access Key
KLING_SECRET_KEY = os.environ.get("KLING_SECRET_KEY", "3kEpKACeCQbrJTEHTHBJafLfbF83eYJC")  # 可灵AI Secret Key

FAL_API_KEY_LIST = FAL_API_KEY.split(",") if FAL_API_KEY else []
GML_API_KEY_LIST = GML_API_KEY.split(",") if GML_API_KEY else []  # 支持多个API KEY轮询
KLING_ACCESS_KEY_LIST = KLING_ACCESS_KEY.split(",") if KLING_ACCESS_KEY else []  # 支持多个可灵Access Key
KLING_SECRET_KEY_LIST = KLING_SECRET_KEY.split(",") if KLING_SECRET_KEY else []  # 支持多个可灵Secret Key

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
    "kontext": {
        "submit_url": "https://queue.fal.run/fal-ai/flux-pro/kontext/max",
        "status_base_url": "https://queue.fal.run/fal-ai/flux-pro"
    },
    "cogview-4-250304": {
        "provider": "zhipuai",  # 标记为智谱提供商
        "model": "cogview-4-250304"  # 智谱模型名称
    },
    "kling-v1-5": {
        "provider": "kling",  # 标记为可灵提供商
        "model": "kling-v1-5"  # 可灵模型名称
    }
}

def make_request(api_key: str, prompt: str):
    url = "https://api.oaiopen.cn/v1/chat/completions"
    models = ["grok-3-beta-flux"]
    model = random.choice(models)
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
    print(f"『执行』: 获取到Flux payload：{payload}")
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
#kontext模型只翻译提示词不做扩展
def make_request2(api_key: str, prompt: str):
    url = "https://api.oaiopen.cn/v1/chat/completions"
    models = ["grok-3-beta-kontext"]
    model = random.choice(models)
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
    print(f"『执行』: 获取到Flux payload：{payload}")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_text = response.text
        print(f"『执行』: 获取到Flux response_text:{response_text}")
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

def get_kling_credentials():
    # 随机获取可灵AI认证凭据
    if KLING_ACCESS_KEY_LIST and KLING_SECRET_KEY_LIST:
        index = random.randint(0, min(len(KLING_ACCESS_KEY_LIST), len(KLING_SECRET_KEY_LIST)) - 1)
        return KLING_ACCESS_KEY_LIST[index], KLING_SECRET_KEY_LIST[index]
    raise ValueError("KLING_ACCESS_KEY or KLING_SECRET_KEY is not set.")

def download_and_encode_image(image_url: str) -> str:
    """
    下载图片并转换为base64格式，或从data URL中提取base64数据
    
    Args:
        image_url: 图片URL或data URL
        
    Returns:
        base64编码的图片数据
    """
    try:
        print(f"正在处理图片: {image_url[:100]}...")
        
        # 检查是否是data URL格式
        if image_url.startswith('data:'):
            print("检测到data URL格式，直接提取base64数据")
            # data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA...
            if 'base64,' in image_url:
                # 提取base64部分
                base64_data = image_url.split('base64,', 1)[1]
                print(f"从data URL提取base64数据成功，长度: {len(base64_data)}")
                return base64_data
            else:
                raise ValueError("data URL格式不正确，缺少base64数据")
        
        # 如果是普通的HTTP/HTTPS URL，则下载图片
        print("检测到HTTP URL，开始下载图片")
        proxies = get_proxies()
        
        # 下载图片
        response = requests.get(
            image_url, 
            proxies=proxies,
            timeout=30,
            verify=False
        )
        response.raise_for_status()
        
        # 转换为base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"图片下载并转换为base64成功，长度: {len(image_base64)}")
        
        return image_base64
        
    except Exception as e:
        print(f"处理图片失败: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")

def generate_kling_jwt_token(access_key: str, secret_key: str, exp_seconds: int = 1800) -> str:
    """
    生成可灵AI的JWT token
    
    Args:
        access_key: Access Key
        secret_key: Secret Key  
        exp_seconds: token有效时间（秒），默认1800秒（30分钟）
    
    Returns:
        生成的JWT token字符串
    """
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": access_key,
        "exp": int(time.time()) + exp_seconds,  # 有效时间
        "nbf": int(time.time()) - 5  # 开始生效的时间，当前时间-5秒
    }
    token = jwt.encode(payload, secret_key, headers=headers)
    return token

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
    
    # 如果是kontext模型，添加默认参数
    if model == "kontext":
        fal_request["guidance_scale"] = options.get("guidance_scale", 3.5)
        fal_request["safety_tolerance"] = options.get("safety_tolerance", "2")
        fal_request["output_format"] = options.get("output_format", "jpeg")
        
        # 添加image_url，优先使用用户提供的
        if "image_url" in options:
            fal_request["image_url"] = options["image_url"]
            print(f"『执行』: 使用用户提供的image_url: {options['image_url']}")
        else:
            # 如果用户没有提供，使用默认值
            fal_request["image_url"] = "http://ai.oaiopen.cn/api/draw/proxy/dd/aHR0cDovL2FpLm9haW9wZW4uY24vZmlsZS9kcmF3L2ZsdXgvZjFjNWUzM2ViZGZmMTZiNDA4MTJiYmVlZDBiMTg0ODYucG5n.png"
            print("『执行』: 使用默认image_url")

    # 添加其他可选参数
    if "seed" in options:
        fal_request["seed"] = options["seed"]
    if "output_format" in options:
        fal_request["output_format"] = options["output_format"]
    
    # 处理图生图模型的特殊参数 - 添加kontext模型的支持
    if model == "kontext" and "image_url" in options:
        fal_request["image_url"] = options["image_url"]
        # 添加其他kontext特有参数
        if "guidance_scale" in options:
            fal_request["guidance_scale"] = options["guidance_scale"]
        if "safety_tolerance" in options:
            fal_request["safety_tolerance"] = options["safety_tolerance"]

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

def call_kling_api(prompt, model, options=None):
    """
    调用可灵AI生成图像
    
    Args:
        prompt: 用于生成图像的文本提示
        model: 使用的模型名称
        options: 附加选项，如aspect_ratio等
        
    Returns:
        成功时返回图像URL列表，失败时抛出异常
    """
    if options is None:
        options = {}
    
    # 可灵AI支持的标准宽高比
    supported_ratios = {
        "1:1": "1:1",      # 正方形
        "16:9": "16:9",    # 横屏
        "9:16": "9:16",    # 竖屏
        "3:4": "3:4",      # 竖屏
        "4:3": "4:3",      # 横屏
        "21:9": "21:9",    # 超宽屏
        "9:21": "9:21"     # 超高屏
    }
    
    # 将size转换为可灵AI支持的aspect_ratio格式
    aspect_ratio = "9:16"  # 默认值
    if "size" in options:
        width, height = map(int, options["size"].split("x"))
        
        # 计算比例并找到最接近的支持比例
        ratio = width / height
        best_ratio = "9:16"  # 默认
        min_diff = float('inf')
        
        for ratio_key, ratio_value in supported_ratios.items():
            w, h = map(int, ratio_key.split(":"))
            target_ratio = w / h
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_ratio = ratio_value
        
        aspect_ratio = best_ratio
        print(f"原始尺寸: {width}x{height}, 转换为可灵AI支持的宽高比: {aspect_ratio}")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # 获取可灵AI认证凭据
            access_key, secret_key = get_kling_credentials()
            authorization = generate_kling_jwt_token(access_key, secret_key)
            
            print(f"Attempt {retry_count + 1}/{max_retries + 1} - Using Kling access_key: {access_key[:5]}...{access_key[-5:] if len(access_key) > 10 else ''}")
            
            # 第一步：提交任务
            url = "https://api-beijing.klingai.com/v1/images/generations"
            payload = {
                "model_name": model,
                "prompt": prompt,
                "n": options.get("num_images", 1),
                "aspect_ratio": aspect_ratio
            }
            
            # 如果有图片URL，下载并转换为base64，然后添加相关字段（图生图功能）
            if "image_url" in options and options["image_url"]:
                try:
                    # 下载图片并转换为base64
                    image_base64 = download_and_encode_image(options["image_url"])
                    payload["image"] = image_base64
                    payload["image_reference"] = "subject"  # 图生图模式必需参数
                    payload["resolution"] = "1k"  # 图生图模式必需参数
                    print(f"可灵AI图生图模式，添加image字段 (base64长度: {len(image_base64)})")
                    print(f"可灵AI图生图模式，添加image_reference字段: subject")
                    print(f"可灵AI图生图模式，添加resolution字段: 1k")
                except Exception as e:
                    print(f"处理图片失败: {str(e)}")
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"图片处理失败，进行重试 ({retry_count}/{max_retries})")
                        time.sleep(2 ** retry_count)
                        continue
                    raise ValueError(f"Failed to process image: {str(e)}")
            
            headers = {
                'Authorization': f'Bearer {authorization}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json',
                'Accept': '*/*',
                'Host': 'api-beijing.klingai.com',
                'Connection': 'keep-alive'
            }
            
            proxies = get_proxies()
            
            print(f"提交可灵AI任务: {prompt[:50]}...")
            submit_response = requests.post(
                url, 
                headers=headers, 
                json=payload,
                proxies=proxies,
                timeout=60,
                verify=False
            )
            
            if submit_response.status_code != 200:
                error_text = submit_response.text
                print(f"可灵AI提交失败: {submit_response.status_code}, {error_text}")
                
                if submit_response.status_code in (401, 403):
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"可灵AI认证失败，尝试使用新的认证凭据重试 ({retry_count}/{max_retries})")
                        time.sleep(2 ** retry_count)
                        continue
                    else:
                        raise ValueError(f"Authentication error with Kling AI after {max_retries} retries: {error_text}")
                
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"可灵AI API错误，进行重试 ({retry_count}/{max_retries})")
                    time.sleep(2 ** retry_count)
                    continue
                
                raise ValueError(f"Kling AI API error: {error_text}")
            
            # 解析提交响应
            submit_data = submit_response.json()
            if submit_data.get('code') != 0:
                error_msg = submit_data.get('message', '未知错误')
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"可灵AI API返回错误，进行重试 ({retry_count}/{max_retries}): {error_msg}")
                    time.sleep(2 ** retry_count)
                    continue
                raise ValueError(f"Kling AI API error: {error_msg}")
            
            task_id = submit_data['data']['task_id']
            print(f"可灵AI任务提交成功，task_id: {task_id}")
            
            # 第二步：轮询任务状态
            max_wait_time = 300  # 5分钟超时
            poll_interval = 5    # 5秒轮询间隔
            start_time = time.time()
            check_count = 0
            
            while time.time() - start_time < max_wait_time:
                check_count += 1
                print(f"第{check_count}次检查可灵AI任务状态: {task_id}")
                
                status_url = f"https://api-beijing.klingai.com/v1/images/generations/{task_id}"
                status_response = requests.get(
                    status_url,
                    headers=headers,
                    proxies=proxies,
                    timeout=60,
                    verify=False
                )
                
                if status_response.status_code != 200:
                    print(f"状态查询失败: {status_response.status_code}")
                    time.sleep(poll_interval)
                    continue
                
                status_data = status_response.json()
                if status_data.get('code') != 0:
                    print(f"状态查询API错误: {status_data.get('message', '未知错误')}")
                    time.sleep(poll_interval)
                    continue
                
                task_data = status_data['data']
                task_status = task_data.get('task_status')
                task_status_msg = task_data.get('task_status_msg', '')
                
                print(f"可灵AI当前状态: {task_status}")
                if task_status_msg:
                    print(f"状态信息: {task_status_msg}")
                
                if task_status == 'succeed':
                    print("🎉 可灵AI任务完成！")
                    # 提取图片URLs
                    image_urls = []
                    if 'task_result' in task_data and 'images' in task_data['task_result']:
                        images = task_data['task_result']['images']
                        for img in images:
                            if 'url' in img:
                                image_urls.append(img['url'])
                                print(f"可灵AI生成图片: {img['url']}")
                    
                    if image_urls:
                        return image_urls
                    else:
                        raise ValueError("可灵AI任务完成但未找到图片")
                        
                elif task_status == 'failed':
                    error_msg = f"可灵AI任务执行失败: {task_status_msg}"
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"{error_msg}，进行重试 ({retry_count}/{max_retries})")
                        time.sleep(2 ** retry_count)
                        break  # 跳出状态检查循环，重新开始外部尝试
                    raise ValueError(error_msg)
                elif task_status in ['submitted', 'processing']:
                    print(f"⏳ 可灵AI任务进行中，{poll_interval}秒后重新检查...")
                    time.sleep(poll_interval)
                else:
                    print(f"⚠️ 未知状态: {task_status}，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
            
            # 超时处理
            if retry_count < max_retries:
                retry_count += 1
                print(f"可灵AI任务等待超时，进行重试 ({retry_count}/{max_retries})")
                time.sleep(2 ** retry_count)
                continue
            
            raise ValueError(f"可灵AI任务等待超时 ({max_wait_time}秒，检查了{check_count}次)")
            
        except ValueError:
            # ValueError是业务逻辑错误，直接抛出
            raise
        except Exception as e:
            error_message = str(e)
            print(f"可灵AI异常: {error_message}")
            
            if retry_count < max_retries:
                retry_count += 1
                print(f"发生异常，进行重试 ({retry_count}/{max_retries}): {error_message}")
                time.sleep(2 ** retry_count)
                continue
            
            raise ValueError(f"Error calling Kling AI: {error_message}")

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
    # 将模型名称转换为小写
    model = model.lower()
    print("『执行』: 当前使用模型为：" + model)
    
    # 获取模型信息
    model_info = MODEL_URLS.get(model)
    if not model_info:
        raise ValueError(f"Unsupported model: {model}")
    
    # 根据提供商分发到不同的API调用函数
    provider = model_info.get("provider")
    if provider == "zhipuai":
        return call_gml_api(prompt, model_info.get("model"), options)
    elif provider == "kling":
        return call_kling_api(prompt, model_info.get("model"), options)
    else:
        return call_fal_api(prompt, model, options)

def create_response(model: str, content: str, include_usage: bool = False, prompt: str = ""):
    """
    创建标准的OpenAI格式响应
    
    Args:
        model: 模型名称
        content: 响应内容
        include_usage: 是否包含使用量信息
        prompt: 原始提示词，用于计算token使用量
    """
    current_time = int(time.time())
    request_id = f"chatcmpl-{current_time}"
    
    # 计算使用量
    usage = {
        "prompt_tokens": len(prompt) // 4 if prompt else 0,
        "completion_tokens": len(content) // 4,
        "total_tokens": (len(prompt) // 4 if prompt else 0) + (len(content) // 4)
    }
    
    # 构建完整的响应
    complete_response = {
        "id": request_id,
        "object": "chat.completion",
        "created": current_time,
        "model": model,
        "system_fingerprint": f"fp_{current_time}",
        "choices": [
            {
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }
        ]
    }
    
    if include_usage:
        complete_response["usage"] = usage
    
    return complete_response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    处理聊天完成请求，路由到不同的模型。
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
    
    # 获取流式选项
    stream_options = openai_request.get('stream_options', {})
    include_usage = stream_options.get('include_usage', False) if stream_options else False

    # 使用最后一条用户消息作为提示
    prompt = ""
    image_url_for_kling = ""  # 用于可灵AI的图片URL
    last_user_message = next((msg['content'] for msg in reversed(messages) if msg.get('role') == 'user'), None)
    if last_user_message:
        # 检查content是否为数组格式（新的视觉模型格式）
        if isinstance(last_user_message, list):
            # 新格式：content是数组，包含text和image_url对象
            for item in last_user_message:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        prompt = item.get('text', '')
                        print(f"『执行』: 从数组格式中提取到文本: {prompt}")
                    elif item.get('type') == 'image_url':
                        # 从数组格式中提取图片URL，用于可灵AI图生图
                        image_url_obj = item.get('image_url', {})
                        if isinstance(image_url_obj, dict):
                            image_url_for_kling = image_url_obj.get('url', '')
                            if image_url_for_kling:
                                print(f"『执行』: 提取到可灵AI图生图URL: {image_url_for_kling}")
        else:
            # 原格式：content是字符串
            prompt = last_user_message
    
    print("『执行』: 用户发送的【原】提示词为：" + prompt)
    
    # 检查是否是kontext模型的图生图请求
    is_img2img = False
    image_url = ""
    guidance_scale = 3.5
    safety_tolerance = "5"
    output_format = "jpeg"
    num_images = 1
    aspect_ratio = "16:9"
    
    # 从请求体中提取图生图参数
    if model == "kontext":
        # 如果是kontext模型，则从消息或请求体中提取图生图参数
        image_url = openai_request.get('image_url', '')
        
        # 如果请求体中没有image_url，尝试从最后一条用户消息对象中直接获取image_url字段
        if not image_url and messages:
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    # 检查新格式：content数组中的image_url
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        # 新格式：从content数组中提取image_url
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                image_url_obj = item.get('image_url', {})
                                if isinstance(image_url_obj, dict):
                                    image_url = image_url_obj.get('url', '')
                                    if image_url:
                                        print(f"『执行』: 从数组格式中获取到image_url: {image_url}")
                                        break
                    else:
                        # 原格式：直接从消息对象中获取image_url字段
                        if 'image_url' in msg:
                            image_url = msg.get('image_url', '')
                            print(f"『执行』: 从消息对象中获取到image_url: {image_url}")
                    
                    if image_url:
                        break
        
        # 如果还没有找到image_url，尝试从最后一条用户消息内容中解析JSON（向后兼容）
        if not image_url and last_user_message and isinstance(last_user_message, str):
            # 尝试从消息中解析JSON
            try:
                import json
                import re
                
                # 尝试找到JSON对象
                json_match = re.search(r'\{.*\}', last_user_message, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        msg_json = json.loads(json_str)
                        if 'image_url' in msg_json:
                            image_url = msg_json.get('image_url', '')
                            prompt = msg_json.get('prompt', prompt)
                            guidance_scale = msg_json.get('guidance_scale', guidance_scale)
                            safety_tolerance = msg_json.get('safety_tolerance', safety_tolerance)
                            aspect_ratio = msg_json.get('aspect_ratio', aspect_ratio)
                            output_format = msg_json.get('output_format', output_format)
                            num_images = msg_json.get('num_images', num_images)
                            print(f"『执行』: 从JSON格式中获取到image_url: {image_url}")
                    except:
                        pass
            except:
                pass
        
        # 提取其他图生图参数
        if not guidance_scale:
            guidance_scale = openai_request.get('guidance_scale', 3.5)
        if not safety_tolerance:
            safety_tolerance = openai_request.get('safety_tolerance', '2')
        if not aspect_ratio:
            aspect_ratio = openai_request.get('aspect_ratio', '16:9')
        if not output_format:
            output_format = openai_request.get('output_format', 'jpeg')
        if not num_images:
            num_images = openai_request.get('num_images', 1)
        
        is_img2img = bool(image_url)
        
        if is_img2img:
            print(f"『执行』: kontext图生图模式，图片URL: {image_url}")
    
    # 针对智谱AI和可灵AI模型处理 - 直接使用原始提示词而不做转换
    if model == "cogview-4-250304":
        print("『执行』: 智谱模型使用原始提示词")
    elif model == "kling-v1-5":
        print("『执行』: 可灵AI模型使用原始提示词")
    elif model == "kontext":
        # 非智谱模型，使用转换后的提示词
        try:
            #调用翻译接口
            converted_prompt = make_request2('sk-Zjjy2zjicDpAh8ge705bCc6a1582406a8dAa88D1E2C9796f', prompt)
            if converted_prompt:
                prompt = converted_prompt
                print("『执行』: 用户发送的【新】提示词为：" + prompt)
        except Exception as e:
            print(f"『执行』: 提示词转换失败: {str(e)}")
        if is_img2img:
            print(f"『执行』: kontext图生图模式使用原始提示词，图片URL: {image_url}")
        else:
            print(f"『执行』: kontext文生图模式使用原始提示词")
    else:
        # 非智谱模型，使用转换后的提示词
        try:
            converted_prompt = make_request('sk-Zjjy2zjicDpAh8ge705bCc6a1582406a8dAa88D1E2C9796f', prompt)
            if converted_prompt:
                prompt = converted_prompt
                print("『执行』: 用户发送的【新】提示词为：" + prompt)
        except Exception as e:
            print(f"『执行』: 提示词转换失败: {str(e)}")
    
    # 检查是否是可灵AI图生图模式
    is_kling_img2img = model == "kling-v1-5" and image_url_for_kling
    
    if not prompt and not is_img2img and not is_kling_img2img:
        # 如果没有提示词且不是图生图模式，返回默认响应
        return jsonify(create_response(
            model=model,
            content="I can generate images. Describe what you'd like.",
            include_usage=include_usage
        ))

    try:
        # 准备选项参数 - 针对不同模型使用不同的默认尺寸
        if model == "kling-v1-5":
            # 可灵AI使用标准的9:16竖屏比例
            default_size = "1080x1920"
        else:
            # 其他模型使用原默认尺寸
            default_size = "1184x880"
            
        options = {
            "size": default_size,
            "seed": 42,
            "output_format": output_format,
            "num_images": num_images
        }
        
        # 为可灵AI模型添加图生图参数
        if model == "kling-v1-5" and image_url_for_kling:
            options["image_url"] = image_url_for_kling
            print(f"『执行』: 可灵AI图生图模式，使用图片: {image_url_for_kling}")
            
        # 为kontext模型添加图生图参数
        if is_img2img:
            options["image_url"] = image_url
            options["guidance_scale"] = guidance_scale
            options["safety_tolerance"] = safety_tolerance
            
        # 调用API生成图像 - 使用统一入口
        image_urls = call_model_api(prompt, model, options)

    # 构建响应内容
        content = ""
        for i, url in enumerate(image_urls):
            if i > 0:
                content += "\n\n"
            content += f"![Generated Image {i + 1}]({url}) "

        if stream:
            # 流式输出
            def generate():
                import json  # 在嵌套函数内部导入json模块避免作用域问题
                current_time = int(time.time())
                request_id = f"chatcmpl-{current_time}"

                # 首先发送角色
                response_role = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "system_fingerprint": f"fp_{current_time}",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_role)}\n\n"

                # 然后发送图片内容
                response_content = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "system_fingerprint": f"fp_{current_time}",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(response_content)}\n\n"

                # 发送完成标记
                response_end = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "system_fingerprint": f"fp_{current_time}",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(response_end)}\n\n"

                # 如果需要包含usage信息
                if include_usage:
                    usage_response = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": current_time,
                        "model": model,
                        "system_fingerprint": f"fp_{current_time}",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(prompt) // 4 if prompt else 0,
                            "completion_tokens": len(content) // 4,
                            "total_tokens": (len(prompt) // 4 if prompt else 0) + (len(content) // 4)
                        }
                    }
                    yield f"data: {json.dumps(usage_response)}\n\n"

                # 结束标记
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
        # 处理API相关的已知错误
        error_message = str(e)
        print(f"『错误』: API调用失败: {error_message}")
        
        # 检查是否是认证错误
        if "Authentication error" in error_message or "api_key" in error_message.lower() or "unauthorized" in error_message.lower():
            return jsonify({
                "error": {
                    "message": error_message,
                    "type": "invalid_api_key"
                }
            }), 401
        
        # 其他API错误
        return jsonify({
            "error": {
                "message": error_message,
                "type": "api_error"
            }
        }), 500

    except Exception as e:
        # 处理未知异常
        error_message = f"生成图像时发生错误: {str(e)}"
        print(f"『错误』: {error_message}")
        return jsonify({
            "error": {
                "message": error_message,
                "type": "server_error"
            }
        }), 500

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
    model = openai_request.get('model', 'flux-dev')
    
    # 对于kontext和可灵AI图生图模型，prompt可以为空
    image_url_direct = openai_request.get('image_url', '')  # 直接从请求中获取image_url
    if not prompt and model not in ["kontext", "kling-v1-5"]:
        return jsonify({
            "error": {
                "message": "prompt is required",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400

    # 提取请求参数 - 针对不同模型使用不同的默认尺寸
    if model == "kling-v1-5":
        default_size = "1080x1920"  # 可灵AI使用9:16标准比例
    else:
        default_size = "1080x1920"  # 保持原有默认值
        
    size = openai_request.get('size', default_size)
    seed = openai_request.get('seed', openai_request.get('user', 100010))
    output_format = openai_request.get('response_format', openai_request.get('output_format', 'jpeg'))
    num_images = openai_request.get('n', openai_request.get('num_images', 1))
    
    # 提取kontext模型特有参数
    image_url = openai_request.get('image_url', '')
    guidance_scale = openai_request.get('guidance_scale', 3.5)
    safety_tolerance = openai_request.get('safety_tolerance', '2')
    
    # 打印原始提示词
    print("『执行』: 图像生成原始提示词：" + prompt)
    
    # 针对非智谱AI和非可灵AI模型的普通文生图，转换提示词
    if model not in ["cogview-4-250304", "kling-v1-5"] and (model != "kontext" or not image_url):
        try:
            # 尝试转换提示词
            converted_prompt = make_request('sk-OUISlfp3DZsJNRaV89676536131e43A88fBd61A80b7739C6', prompt)
            if converted_prompt:
                prompt = converted_prompt
                print("『执行』: 图像生成转换后提示词：" + prompt)
        except Exception as e:
            print(f"『执行』: 提示词转换失败: {str(e)}")
    else:
        if model == "cogview-4-250304":
            print("『执行』: 智谱模型使用原始提示词")
        elif model == "kling-v1-5":
            print("『执行』: 可灵AI模型使用原始提示词")
        else:
            print(f"『执行』: kontext图生图模式，使用原始提示词，图片URL: {image_url}")

    # 准备选项参数
    options = {
        "size": size,
        "seed": seed,
        "output_format": output_format,
        "num_images": num_images
    }
    
    # 为可灵AI模型添加图生图参数
    if model == "kling-v1-5" and image_url_direct:
        options["image_url"] = image_url_direct
        print(f"『执行』: 可灵AI图生图模式（直接端点），使用图片: {image_url_direct}")
        
    # 为kontext模型添加图生图参数
    if model == "kontext" and image_url:
        options["image_url"] = image_url
        options["guidance_scale"] = guidance_scale
        options["safety_tolerance"] = safety_tolerance

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
        {"id": "kontext", "object": "model", "created": 1698785189,
         "owned_by": "fal-openai-adapter", "permission": [], "root": "kontext", "parent": None},
        {"id": "cogview-4-250304", "object": "model", "created": 1698785189,
         "owned_by": "zhipuai-adapter", "permission": [], "root": "cogview-4-250304", "parent": None},
        {"id": "kling-v1-5", "object": "model", "created": 1698785189,
         "owned_by": "kling-adapter", "permission": [], "root": "kling-v1-5", "parent": None}
    ]
    return jsonify({"object": "list", "data": models})


if __name__ == "__main__":
    # 检查环境变量
    missing_keys = []
    if not FAL_API_KEY:
        print("警告: FAL_API_KEY 未设置。FAL模型功能将不可用。")
        missing_keys.append("FAL_API_KEY")
    
    if not GML_API_KEY and ZhipuAI is not None:
        print("警告: GML_API_KEY 未设置。智谱文生图功能将不可用。")
        missing_keys.append("GML_API_KEY")
    
    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        print("警告: KLING_ACCESS_KEY 或 KLING_SECRET_KEY 未设置。可灵AI功能将不可用。")
        missing_keys.append("KLING_CREDENTIALS")
    
    if missing_keys:
        print(f"缺少以下环境变量/配置: {', '.join(missing_keys)}")
        # 注释掉严格的检查，允许部分功能运行
        # if "FAL_API_KEY" in missing_keys:
        #     raise ValueError("FAL_API_KEY is not set.")

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
