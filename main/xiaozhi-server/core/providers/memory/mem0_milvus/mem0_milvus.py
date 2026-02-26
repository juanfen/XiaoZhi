import os
# 选择第3张卡（从0开始计数），必须在任何 torch/transformers 导入或模型加载之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from ..base import MemoryProviderBase, logger
from mem0 import Memory

TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory=None):
        super().__init__(config)


        try:
            def clean_config(config: dict) -> dict:
                import ast
                cleaned = {}
                for k, v in config.items():
                    if isinstance(v, str):
                        try:
                            parsed = ast.literal_eval(v)
                            if isinstance(parsed, dict):
                                cleaned[k] = parsed
                            else:
                                cleaned[k] = v
                        except:
                            cleaned[k] = v
                    else:
                        cleaned[k] = v
                return cleaned

            self.config = clean_config(self.config)

            self.client = Memory.from_config(self.config)
            self.use_mem0local = True
            logger.bind(tag=TAG).info("成功使用Mem0_milvus服务")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Mem0配置错误: {str(e)}")
            self.use_mem0local = False

    #实时存储
    # async def save_memory(self, msgs):
    async def save_memory(self, msgs,user_name=None):
        logger.bind(tag=TAG).debug(f"Mem0服务状态: {self.use_mem0local}")
        print(f"存储记忆用户:{user_name}")
        if not self.use_mem0local:
            return None
        # if len(msgs) < 2:
        #     return None

        try:
            # 兼容字典和对象的消息访问
            valid_messages = []
            for message in msgs:
                # 统一访问方式 - 兼容字典和对象
                role = message.get('role') if hasattr(message, 'get') else getattr(message, 'role', '')
                content = message.get('content') if hasattr(message, 'get') else getattr(message, 'content', None)

                # 有效性检查
                if (
                        role != "system" and
                        content is not None and
                        isinstance(content, str) and
                        content.strip() != ""  # 非空字符串
                ):
                    valid_messages.append({
                        "role": role,
                        "content": content
                    })

            # 增强日志记录（保持原样）
            invalid_count = len(msgs) - len(valid_messages)
            if invalid_count > 0:
                logger.info(f"过滤掉 {invalid_count} 条无效消息")

            # 检查有效消息数量（保持原样）
            # if len(valid_messages) < 2:
            #     logger.debug("有效消息不足，跳过保存")
            #     return None

            # 转换格式（根据Mem0要求）
            messages = [
                {"role": msg['role'], "content": msg['content']}
                for msg in valid_messages
            ]

            from datetime import datetime
            from zoneinfo import ZoneInfo

            bj = ZoneInfo("Asia/Shanghai")
            now = datetime.now(bj)  # 带时区的北京时间
            print(now.strftime("%Y-%m-%d %H:%M"))  # 例如：2025-12-20 10:36


            time = now.strftime("%Y-%m-%d %H:%M")
            run_id = self.role_id
            print(f"存储设备id:{run_id}")
            # 原始存储逻辑（完全保持不变）
            result = self.client.add(messages,run_id=self.role_id,user_id=user_name,metadata = {"created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            # metadata = {"created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            # result = self.client.add(messages, user_id=user_name)
            # logger.bind(tag=TAG).info(f"add method return type: {type(result)}, value: {result}")
            #   logger.info(f"成功保存 {len(valid_messages)} 条记忆")
            logger.bind(tag=TAG).info(f"[add] user_id={user_name}Save memory result: {result}")

        except Exception as e:
            print(e)
            logger.bind(tag=TAG).error(f"保存记忆失败: {str(e)}")
            return None


    async def query_memory(self, query: str,user_name=None) -> str:
        # print(f"查询用户名：{user_name}")
        if not self.use_mem0local:
            return ""
        try:
            run_id = self.role_id
            print(run_id)
            results = self.client.search(
                query,
                user_id=user_name,# user_id=self.role_id,
                agent_id= None,
                run_id= self.role_id,
                limit = 5,
                filters=None,
                # filters={
                #     "run_id": self.role_id  # 添加 filters 进行严格过滤
                # }
            )
            # logger.bind(tag=TAG).info(f"get memory result: {results}")
            if not results or 'results' not in results:
                return ""

            # Format each memory entry with its update time up to minutes
            memories = []
            for entry in results['results']:
                metadata = entry.get('metadata', {})
                if metadata == None:
                   timestamp = entry.get('updated_at', '') or entry.get('created_at', '')
                else:
                    timestamp = metadata.get('created_time', '') or entry.get('updated_at', '')
                # timestamp = entry.get('updated_at', '') or entry.get('created_at', '')
                if timestamp:
                    try:
                        # Parse and reformat the timestamp
                        dt = timestamp.split('.')[0]  # Remove milliseconds
                        formatted_time = dt.replace('T', ' ')
                    except:
                        formatted_time = timestamp
                memory = entry.get('memory', '')
                if timestamp and memory:
                    # Store tuple of (timestamp, formatted_string) for sorting

                    memories.append((timestamp, f"[{formatted_time}{memory}]"))

            # Sort by timestamp in descending order (newest first)
            memories.sort(key=lambda x: x[0], reverse=True)

            # Extract only the formatted strings
            memories_str = "\n".join(f"- {memory[1]}" for memory in memories)

            logger.bind(tag=TAG).info(f"记忆查询结果: {memories_str}")
            return memories_str
        except Exception as e:
            if "collection not found" in e:
                return ""
            else:
                logger.bind(tag=TAG).error(f"查询记忆失败: {str(e)}")
            return ""
    # async def __delete__(self, instance):
    async def delete_all_memories(self, user_name = None) -> bool:
        """
        清空指定用户的所有记忆
        :param user_name: 用户ID，默认为当前实例的role_id
        :return: 成功返回True，失败返回False
        """
        if not self.use_mem0local:
            logger.bind(tag=TAG).warning("Mem0服务未启用，无法清空记忆")
            return False

        user_name = user_name if user_name is not None else self.role_id

        try:

            result = self.client.delete_all(user_id=user_name,run_id= self.role_id)
            logger.bind(tag=TAG).info(f"已清空用户 {user_name} 的所有记忆: {result}")
            return True
        except Exception as e:
            logger.bind(tag=TAG).error(f"清空记忆失败: {str(e)}")
            return False

