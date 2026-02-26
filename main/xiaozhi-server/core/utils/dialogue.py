import uuid
import re
from typing import List, Dict
from datetime import datetime
import os

import json
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from config.logger import setup_logging
logger = setup_logging()

class Message:
    def __init__(
        self,
        role: str,
        content: str = None,
        uniq_id: str = None,
        tool_calls=None,
        tool_call_id=None,
    ):
        self.uniq_id = uniq_id if uniq_id is not None else str(uuid.uuid4())
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class Dialogue:
    def __init__(self):
        self.dialogue: List[Message] = []
        # 获取当前时间
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def put(self, message: Message):
        self.dialogue.append(message)

    def getMessages(self, m, dialogue):
        if m.tool_calls is not None:
            dialogue.append({"role": m.role, "tool_calls": m.tool_calls})
        elif m.role == "tool":
            dialogue.append(
                {
                    "role": m.role,
                    "tool_call_id": (
                        str(uuid.uuid4()) if m.tool_call_id is None else m.tool_call_id
                    ),
                    "content": m.content,
                }
            )
        else:
            dialogue.append({"role": m.role, "content": m.content})

    def get_llm_dialogue(self) -> List[Dict[str, str]]:
        # 直接调用get_llm_dialogue_with_memory，传入None作为memory_str
        # 这样确保说话人功能在所有调用路径下都生效
        return self.get_llm_dialogue_with_memory(None, None)

    def update_system_message(self, new_content: str):
        """更新或添加系统消息"""
        # 查找第一个系统消息
        system_msg = next((msg for msg in self.dialogue if msg.role == "system"), None)
        if system_msg:
            system_msg.content = new_content
        else:
            self.put(Message(role="system", content=new_content))

    # 记忆修改 message拼接 用户画像+长期记忆片段+摘要+短期上下文
    def get_llm_dialogue_with_memory(
        self, memory_str: str = None, voiceprint_config: dict = None,user_name = None
    ) -> List[Dict[str, str]]:
        # 构建对话
        dialogue = []

        # 添加系统提示和记忆
        system_message = next(
            (msg for msg in self.dialogue if msg.role == "system"), None
        )

        # 增加摘要读取
        from pathlib import Path
        base = Path("/home/wutianxin@kemande.cn/xiaozhi-esp32-server-updata/main/summary")
        base_2 = Path("/mnt/Data/chenjiahui/User")
        file_path = base / f"{user_name}对话摘要.txt"
        summary_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                summary_text = f.read().strip()
        except FileNotFoundError:
            print(f"找不到摘要文件：{file_path}")
        except Exception as e:
            print(f"读取摘要文件出错：{e}")

        # 增加上下文、用户画像读取
        file_path_1 = base / f"{user_name}近期上下文.txt"
        file_path_2 = base_2 / f"{user_name}用户画像.txt"
        recent_text = ""
        try:
            with open(file_path_1, 'r', encoding='utf-8') as f:
                recent_text = f.read().strip()
        except FileNotFoundError:
            print(f"找不到上下文文件：{file_path_1}")
        except Exception as e:
            print(f"读取上下文文件出错：{e}")
        recent_text_2 = ""
        try:
            with open(file_path_2, 'r', encoding='utf-8') as f:
                recent_text_2 = f.read().strip()
        except FileNotFoundError:
            print(f"找不到用户文件：{file_path_2}")
        except Exception as e:
            print(f"读取用户文件出错：{e}")

        from zoneinfo import ZoneInfo
        current_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        print(current_time)
        if system_message:
            enhanced_system_prompt = (
                f"{system_message.content}"
                f"以下是{user_name}相关信息，{recent_text_2}，当没有明确聊天话题时，可根据其中内容对用户展开聊天与回复。"
                f"以下是用户的历史记忆总结{summary_text}{memory_str}。"
                f"以下是近期对话上下文：{recent_text}"
                f"以下是当前对话时间：{current_time}"

            )
            dialogue.append({"role": "system", "content": enhanced_system_prompt})
            TAG = __name__
            logger.bind(tag=TAG).info("message拼接")
        # if system_message:
        #     # 基础系统提示
        #     enhanced_system_prompt = system_message.content
        #     # 替换时间占位符
        #     enhanced_system_prompt = enhanced_system_prompt.replace(
        #         "{{current_time}}", datetime.now().strftime("%H:%M")
        #     )

            # 添加说话人个性化描述
            try:
                speakers = voiceprint_config.get("speakers", [])
                if speakers:
                    enhanced_system_prompt += "\n\n<speakers_info>"
                    for speaker_str in speakers:
                        try:
                            parts = speaker_str.split(",", 2)
                            if len(parts) >= 2:
                                name = parts[1].strip()
                                # 如果描述为空，则为""
                                description = (
                                    parts[2].strip() if len(parts) >= 3 else ""
                                )
                                enhanced_system_prompt += f"\n- {name}：{description}"
                        except:
                            pass
                    enhanced_system_prompt += "\n\n</speakers_info>"
            except:
                # 配置读取失败时忽略错误，不影响其他功能
                pass

            # # 使用正则表达式匹配 <memory> 标签，不管中间有什么内容
            # if memory_str is not None:
            #     enhanced_system_prompt = re.sub(
            #         r"<memory>.*?</memory>",
            #         f"<memory>\n{memory_str}\n</memory>",
            #         enhanced_system_prompt,
            #         flags=re.DOTALL,
            #     )
            # dialogue.append({"role": "system", "content": enhanced_system_prompt})
        # 添加用户和助手的对话
        for m in self.dialogue:
            if m.role != "system":  # 跳过原始的系统消息
                self.getMessages(m, dialogue)
        # print("message:",dialogue)
        logger.bind(tag=TAG).info("响应开始")
        return dialogue




