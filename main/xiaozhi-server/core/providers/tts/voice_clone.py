import os
import queue
import asyncio
import traceback
import aiohttp
from aiohttp import FormData
import requests
import time
import base64
from config.logger import setup_logging
from core.utils.tts import MarkdownCleaner
from core.providers.tts.base import TTSProviderBase
from core.utils import opus_encoder_utils, textUtils
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.interface_type = InterfaceType.SINGLE_STREAM
        self.api_url = config.get("api_url")
        self.audio_format = "pcm"

        # 使用字典来维护每个session_id的缓冲区
        self.session_buffers = {}  # {session_id: {'text_buff': [], 'processed_chars': 0, 'segment_count': 0}}
        self.before_stop_play_files = []

        # 语音克隆参数
        self.temperature = config.get("temperature", 0.9)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 50)
        self.max_tokens = config.get("max_tokens", 2048)

        # 读取参考音频文件并编码为base64
        self.reference_audio_path = config.get("reference_audio_path")
        self.reference_text = config.get("reference_text")
        self.pitch = config.get("pitch")
        self.speed = config.get("speed")
        self.voice = config.get("private_voice","default")
        # try:
        #     with open(self.reference_audio_path, "rb") as f:
        #         audio_bytes = f.read()
        #     self.reference_audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        # except Exception as e:
        #     logger.bind(tag=TAG).error(f"读取参考音频文件失败: {e}")
        #     raise

        # 创建Opus编码器
        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=16000, channels=1, frame_size_ms=60
        )

        # PCM缓冲区
        self.pcm_buffer = bytearray()

    def _get_session_buffer(self, session_id):
        """获取或创建指定session_id的缓冲区"""
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = {
                'text_buff': [],
                'processed_chars': 0,
                'segment_count': 0,
                'tts_audio_first_sentence': True
            }
        return self.session_buffers[session_id]

    def _cleanup_session_buffer(self, session_id):
        """清理指定session_id的缓冲区"""
        if session_id in self.session_buffers:
            del self.session_buffers[session_id]

    ###################################################################################
    # 单流式TTS重写父类的方法--开始
    ###################################################################################

    def tts_text_priority_thread(self):
        """流式文本处理线程"""
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                session_id = self.conn.sentence_id  # 获取连接的session_id
                buffer = self._get_session_buffer(session_id)  # 获取对应的缓冲区

                if message.sentence_type == SentenceType.FIRST:
                    # 初始化当前session的参数
                    self.tts_stop_request = False
                    buffer['text_buff'] = []
                    buffer['processed_chars'] = 0
                    buffer['segment_count'] = 0
                    buffer['tts_audio_first_sentence'] = True
                    self.before_stop_play_files.clear()

                    # 如果是文本内容，添加到缓冲区
                    if ContentType.TEXT == message.content_type:
                        buffer['text_buff'].append(message.content_detail)
                        # 处理所有可分割的文本段
                        while True:
                            segment_text = self._get_segment_text(buffer)
                            if segment_text:
                                self.to_tts_single_stream(segment_text, session_id)
                            else:
                                break
                elif ContentType.TEXT == message.content_type:
                    buffer['text_buff'].append(message.content_detail)
                    # 处理所有可分割的文本段
                    while True:
                        segment_text = self._get_segment_text(buffer)
                        if segment_text:
                            self.to_tts_single_stream(segment_text, session_id)
                        else:
                            break
                elif ContentType.FILE == message.content_type:
                    logger.bind(tag=TAG).info(
                        f"添加音频文件到待播放列表: {message.content_file}"
                    )
                    if message.content_file and os.path.exists(message.content_file):
                        # 先处理文件音频数据
                        file_audio = self._process_audio_file(message.content_file)
                        self.before_stop_play_files.append(
                            (file_audio, message.content_detail)
                        )
                if message.sentence_type == SentenceType.LAST:
                    # 处理剩余的文本
                    self._process_remaining_text(buffer, session_id, True)
                    # 清理该session的缓冲区
                    self._cleanup_session_buffer(session_id)

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )

    def _get_segment_text(self, buffer):
        # 定义标点符号集合
        first_sentence_punctuations = [',', '，']  # 第一句话可以使用的标点（逗号）
        punctuations = ['.', '。', '!', '！', '?', '？', ';', '；']  # 后续句子使用的标点

        # 合并当前全部文本
        full_text = "".join(buffer['text_buff'])
        current_text = full_text[buffer['processed_chars']:]  # 从未处理的位置开始

        # 如果没有文本或已经处理完所有文本，返回None
        if not current_text:
            return None

        last_punct_pos = -1

        # 根据是否是第一句话选择不同的标点符号集合
        punctuations_to_use = (
            first_sentence_punctuations
            if buffer.get('tts_audio_first_sentence', True)
            else punctuations
        )

        # 查找标点符号位置
        for punct in punctuations_to_use:
            pos = current_text.rfind(punct)
            # 找到最右边的标点位置
            if pos != -1 and (last_punct_pos == -1 or pos > last_punct_pos):
                last_punct_pos = pos

        if last_punct_pos != -1:
            segment_text_raw = current_text[: last_punct_pos + 1]
            segment_text = textUtils.get_string_no_punctuation_or_emoji(segment_text_raw)
            buffer['processed_chars'] += len(segment_text_raw)  # 更新已处理字符位置

            # 如果是第一句话，在找到第一个逗号后，将标志设置为False
            if buffer.get('tts_audio_first_sentence', True):
                buffer['tts_audio_first_sentence'] = False

            return segment_text
        elif self.tts_stop_request and current_text:
            segment_text = textUtils.get_string_no_punctuation_or_emoji(current_text)
            buffer['tts_audio_first_sentence'] = True  # 重置标志
            return segment_text
        else:
            return None

    def _process_remaining_text(self, buffer, session_id, is_last=False):
        """处理剩余的文本并生成语音"""
        # 处理所有可分割的文本段
        while True:
            segment_text = self._get_segment_text(buffer)
            if segment_text:
                self.to_tts_single_stream(segment_text, session_id, is_last)
            else:
                break

        # 处理文件音频
        self._process_before_stop_play_files()

    def to_tts_single_stream(self, text, session_id, is_last=False):
        try:
            max_repeat_time = 5
            text = MarkdownCleaner.clean_markdown(text)
            try:
                asyncio.run(self.text_to_speak(text, session_id, is_last))
            except Exception as e:
                logger.bind(tag=TAG).warning(
                    f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
                )
                max_repeat_time -= 1

            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"语音生成成功: {text}，重试{5 - max_repeat_time}次"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"语音生成失败: {text}，请检查网络或服务是否正常"
                )
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
        finally:
            return None

    ###################################################################################
    # 单流式TTS重写父类的方法--结束
    ###################################################################################

    async def text_to_speak(self, text, session_id, is_last):
        """流式处理TTS音频，每句只推送一次音频列表"""
        await self._tts_request(text, is_last)

    async def _tts_request(self, text: str, is_last: bool) -> None:
        # 构建语音克隆请求参数，全部转字符串
        payload = {
            "text": text,
            "reference_text": self.reference_text,
            "reference_audio": self.reference_audio_path or "",
            "temperature": str(self.temperature),
            "top_p": str(self.top_p),
            "top_k": str(self.top_k),
            "max_tokens": str(self.max_tokens),
            "stream": "true",  # 必须小写字符串
            "response_format": "pcm",
            "pitch": self.pitch,
            "speed": self.speed,
            "voice": self.voice
        }
        logger.bind(tag=TAG).info(f"TTS请求参数1: {payload['voice']}")

        # 构建表单数据
        form = FormData()
        for k, v in payload.items():
            form.add_field(k, v)

        # 一帧 PCM 所需字节数
        frame_bytes = int(
            self.opus_encoder.sample_rate
            * self.opus_encoder.channels
            * self.opus_encoder.frame_size_ms
            / 1000
            * 2
        )

        pcm_buffer = bytearray()
        opus_datas_cache = []
        segment_count = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=form, timeout=30) as resp:
                    if resp.status != 200:
                        logger.bind(tag=TAG).error(
                            f"TTS请求失败: {resp.status}, {await resp.text()}"
                        )
                        self.tts_audio_queue.put((SentenceType.LAST, [], None))
                        return

                    self.tts_audio_queue.put((SentenceType.FIRST, [], text))

                    # 逐块读取音频数据
                    async for chunk in resp.content.iter_chunked(1024):
                        if not chunk:
                            continue

                        pcm_buffer.extend(chunk)

                        # 每够一帧就编码发送
                        while len(pcm_buffer) >= frame_bytes:
                            frame = bytes(pcm_buffer[:frame_bytes])
                            del pcm_buffer[:frame_bytes]

                            # 添加错误处理
                            try:
                                opus = self.opus_encoder.encode_pcm_to_opus_stream(frame, end_of_stream=False,callback=self.handle_opus)
                                if opus:
                                    if segment_count < 20:
                                        self.tts_audio_queue.put((SentenceType.MIDDLE, opus, None))
                                        segment_count += 1
                                    else:
                                        opus_datas_cache.extend(opus)
                            except Exception as e:
                                logger.bind(tag=TAG).error(f"Opus编码失败: {e}")
                                continue

                    # 处理剩余不足一帧的数据
                    if pcm_buffer:
                        # 确保最后一帧长度正确
                        if len(pcm_buffer) < frame_bytes:
                            # 填充最后一帧
                            padding = b'\x00' * (frame_bytes - len(pcm_buffer))
                            pcm_buffer.extend(padding)

                        try:
                            opus = self.opus_encoder.encode_pcm_to_opus_stream(bytes(pcm_buffer), end_of_stream=True,callback=self.handle_opus)
                            if opus:
                                if segment_count < 20:
                                    self.tts_audio_queue.put((SentenceType.MIDDLE, opus, None))
                                    segment_count += 1
                                else:
                                    opus_datas_cache.extend(opus)
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"最后一帧Opus编码失败: {e}")

                    # 如果不是前10个片段，最后统一发送缓存
                    if segment_count >= 20 and opus_datas_cache:
                        self.tts_audio_queue.put((SentenceType.MIDDLE, opus_datas_cache, None))

                    # 如果是最后一段，输出音频获取完毕
                    if is_last:
                        self._process_before_stop_play_files()
                        # 发送结束标记
                        self.tts_audio_queue.put((SentenceType.LAST, [], None))

        except Exception as e:
            logger.bind(tag=TAG).error(f"TTS请求异常: {e}\n{traceback.format_exc()}")
            self.tts_audio_queue.put((SentenceType.LAST, [], None))

    def to_tts(self, text: str) -> list:
        """非流式TTS处理，用于测试及保存音频文件的场景

        Args:
            text: 要转换的文本

        Returns:
            list: 返回opus编码后的音频数据列表
        """
        start_time = time.time()
        text = MarkdownCleaner.clean_markdown(text)

        # 构建语音克隆请求参数
        payload = {
            "text": text,
            "reference_text": self.reference_text,
            "reference_audio": self.reference_audio_path,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": "pcm",
            "pitch": self.pitch,
            "speed": self.speed,
            "voice": self.voice
        }
        logger.bind(tag=TAG).info(f"TTS请求参数2: {payload['voice']}")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(
                self.api_url, data=payload, headers=headers, timeout=30
            )
            if response.status_code != 200:
                logger.bind(tag=TAG).error(
                    f"TTS请求失败: {response.status_code}, {response.text}"
                )
                return []

            logger.info(f"TTS请求成功: {text}, 耗时: {time.time() - start_time}秒")

            # 使用opus编码器处理PCM数据
            opus_datas = []
            pcm_data = response.content

            # 计算每帧的字节数
            frame_bytes = int(
                self.opus_encoder.sample_rate
                * self.opus_encoder.channels
                * self.opus_encoder.frame_size_ms
                / 1000
                * 2
            )

            # 分帧处理PCM数据
            for i in range(0, len(pcm_data), frame_bytes):
                frame = pcm_data[i: i + frame_bytes]
                if len(frame) < frame_bytes:
                    # 最后一帧可能不足，用0填充
                    frame = frame + b"\x00" * (frame_bytes - len(frame))

                try:
                    opus = self.opus_encoder.encode_pcm_to_opus_stream(
                        frame, end_of_stream=(i + frame_bytes >= len(pcm_data)),callback=lambda opus: opus_datas.append(opus)
                    )
                    if opus:
                        opus_datas.extend(opus)
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Opus编码失败: {e}")
                    continue

            return opus_datas

        except Exception as e:
            logger.bind(tag=TAG).error(f"TTS请求异常: {e}")
            return []
# import os
# import queue
# import asyncio
# import traceback
# import aiohttp
# from aiohttp import FormData
# import requests
# import time
# import base64
# from config.logger import setup_logging
# from core.utils.tts import MarkdownCleaner
# from core.providers.tts.base import TTSProviderBase
# from core.utils import opus_encoder_utils, textUtils
# from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType
#
# TAG = __name__
# logger = setup_logging()
#
#
# class TTSProvider(TTSProviderBase):
#     def __init__(self, config, delete_audio_file):
#         super().__init__(config, delete_audio_file)
#         self.interface_type = InterfaceType.SINGLE_STREAM
#         self.api_url = config.get("api_url")
#         self.audio_format = "pcm"
#
#         # 使用字典来维护每个session_id的缓冲区
#         self.session_buffers = {}  # {session_id: {'text_buff': [], 'processed_chars': 0, 'segment_count': 0}}
#         self.before_stop_play_files = []
#
#         # 语音克隆参数
#         self.temperature = config.get("temperature", 0.9)
#         self.top_p = config.get("top_p", 0.95)
#         self.top_k = config.get("top_k", 50)
#         self.max_tokens = config.get("max_tokens", 2048)
#
#         # 读取参考音频文件并编码为base64
#         self.reference_audio_path = config.get("reference_audio_path")
#         self.reference_text = config.get("reference_text")
#         self.pitch = config.get("pitch")
#         self.speed = config.get("speed")
#         try:
#             with open(self.reference_audio_path, "rb") as f:
#                 audio_bytes = f.read()
#             self.reference_audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
#         except Exception as e:
#             logger.bind(tag=TAG).error(f"读取参考音频文件失败: {e}")
#             raise
#
#         # 创建Opus编码器
#         self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
#             sample_rate=16000, channels=1, frame_size_ms=60
#         )
#
#         # PCM缓冲区
#         self.pcm_buffer = bytearray()
#
#     def _get_session_buffer(self, session_id):
#         """获取或创建指定session_id的缓冲区"""
#         if session_id not in self.session_buffers:
#             self.session_buffers[session_id] = {
#                 'text_buff': [],
#                 'processed_chars': 0,
#                 'segment_count': 0,
#                 'tts_audio_first_sentence': True
#             }
#         return self.session_buffers[session_id]
#
#     def _cleanup_session_buffer(self, session_id):
#         """清理指定session_id的缓冲区"""
#         if session_id in self.session_buffers:
#             del self.session_buffers[session_id]
#
#     ###################################################################################
#     # 单流式TTS重写父类的方法--开始
#     ###################################################################################
#
#     def tts_text_priority_thread(self):
#         """流式文本处理线程"""
#         while not self.conn.stop_event.is_set():
#             try:
#                 message = self.tts_text_queue.get(timeout=1)
#                 session_id = self.conn.session_id  # 获取连接的session_id
#                 buffer = self._get_session_buffer(session_id)  # 获取对应的缓冲区
#
#                 if message.sentence_type == SentenceType.FIRST:
#                     # 初始化当前session的参数
#                     self.tts_stop_request = False
#                     buffer['text_buff'] = []
#                     buffer['processed_chars'] = 0
#                     buffer['segment_count'] = 0
#                     buffer['tts_audio_first_sentence'] = True
#                     self.before_stop_play_files.clear()
#
#                     # 如果是文本内容，添加到缓冲区
#                     if ContentType.TEXT == message.content_type:
#                         buffer['text_buff'].append(message.content_detail)
#                         # 处理所有可分割的文本段
#                         while True:
#                             segment_text = self._get_segment_text(buffer)
#                             if segment_text:
#                                 self.to_tts_single_stream(segment_text, session_id)
#                             else:
#                                 break
#                 elif ContentType.TEXT == message.content_type:
#                     buffer['text_buff'].append(message.content_detail)
#                     # 处理所有可分割的文本段
#                     while True:
#                         segment_text = self._get_segment_text(buffer)
#                         if segment_text:
#                             self.to_tts_single_stream(segment_text, session_id)
#                         else:
#                             break
#                 elif ContentType.FILE == message.content_type:
#                     logger.bind(tag=TAG).info(
#                         f"添加音频文件到待播放列表: {message.content_file}"
#                     )
#                     if message.content_file and os.path.exists(message.content_file):
#                         # 先处理文件音频数据
#                         file_audio = self._process_audio_file(message.content_file)
#                         self.before_stop_play_files.append(
#                             (file_audio, message.content_detail)
#                         )
#                 if message.sentence_type == SentenceType.LAST:
#                     # 处理剩余的文本
#                     self._process_remaining_text(buffer, session_id, True)
#                     # 清理该session的缓冲区
#                     self._cleanup_session_buffer(session_id)
#
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 logger.bind(tag=TAG).error(
#                     f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
#                 )
#
#     def _get_segment_text(self, buffer):
#         # 定义标点符号集合
#         first_sentence_punctuations = [',', '，']  # 第一句话可以使用的标点（逗号）
#         punctuations = ['.', '。', '!', '！', '?', '？', ';', '；']  # 后续句子使用的标点
#
#         # 合并当前全部文本
#         full_text = "".join(buffer['text_buff'])
#         current_text = full_text[buffer['processed_chars']:]  # 从未处理的位置开始
#
#         # 如果没有文本或已经处理完所有文本，返回 None
#         if not current_text:
#             return None
#
#         last_punct_pos = -1
#
#         # 根据是否是第一句话选择不同的标点符号集合
#         punctuations_to_use = (
#             first_sentence_punctuations
#             if buffer.get('tts_audio_first_sentence', True)
#             else punctuations
#         )
#
#         # 查找标点符号位置
#         for punct in punctuations_to_use:
#             pos = current_text.rfind(punct)
#             # 找到最右边的标点位置
#             if pos != -1 and (last_punct_pos == -1 or pos > last_punct_pos):
#                 last_punct_pos = pos
#
#         if last_punct_pos != -1:
#             segment_text_raw = current_text[: last_punct_pos + 1]
#             segment_text = textUtils.get_string_no_punctuation_or_emoji(segment_text_raw)
#             buffer['processed_chars'] += len(segment_text_raw)  # 更新已处理字符位置
#
#             # 如果是第一句话，在找到第 **第二个逗号** 后，将标志设置为 False
#             if buffer.get('tts_audio_first_sentence', True):
#                 comma_count = buffer.get('tts_first_comma_count', 0)
#                 comma_count += segment_text_raw.count(',') + segment_text_raw.count('，')
#                 buffer['tts_first_comma_count'] = comma_count
#                 if comma_count >= 2:  # 第二个逗号出现后
#                     buffer['tts_audio_first_sentence'] = False
#
#             return segment_text
#         elif self.tts_stop_request and current_text:
#             segment_text = textUtils.get_string_no_punctuation_or_emoji(current_text)
#             buffer['tts_audio_first_sentence'] = True  # 重置标志
#             buffer['tts_first_comma_count'] = 0  # 计数器重置
#             return segment_text
#         else:
#             return None
#
#     def _process_remaining_text(self, buffer, session_id, is_last=False):
#         """处理剩余的文本并生成语音"""
#         # 处理所有可分割的文本段
#         while True:
#             segment_text = self._get_segment_text(buffer)
#             if segment_text:
#                 self.to_tts_single_stream(segment_text, session_id, is_last)
#             else:
#                 break
#
#         # 处理文件音频
#         self._process_before_stop_play_files()
#
#     def to_tts_single_stream(self, text, session_id, is_last=False):
#         try:
#             max_repeat_time = 5
#             text = MarkdownCleaner.clean_markdown(text)
#             try:
#                 asyncio.run(self.text_to_speak(text, session_id, is_last))
#             except Exception as e:
#                 logger.bind(tag=TAG).warning(
#                     f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
#                 )
#                 max_repeat_time -= 1
#
#             if max_repeat_time > 0:
#                 logger.bind(tag=TAG).info(
#                     f"语音生成成功: {text}，重试{5 - max_repeat_time}次"
#                 )
#             else:
#                 logger.bind(tag=TAG).error(
#                     f"语音生成失败: {text}，请检查网络或服务是否正常"
#                 )
#         except Exception as e:
#             logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
#         finally:
#             return None
#
#     ###################################################################################
#     # 单流式TTS重写父类的方法--结束
#     ###################################################################################
#
#     async def text_to_speak(self, text, session_id, is_last):
#         """流式处理TTS音频，每句只推送一次音频列表"""
#         await self._tts_request(text, is_last)
#
#     async def _tts_request(self, text: str, is_last: bool) -> None:
#         # 构建语音克隆请求参数，全部转字符串
#         payload = {
#             "text": text,
#             "reference_text": self.reference_text,
#             "reference_audio": self.reference_audio_base64 or "",
#             "temperature": str(self.temperature),
#             "top_p": str(self.top_p),
#             "top_k": str(self.top_k),
#             "max_tokens": str(self.max_tokens),
#             "stream": "true",  # 必须小写字符串
#             "response_format": "pcm",
#             "pitch": self.pitch,
#             "speed": self.speed
#         }
#
#         # 构建表单数据
#         form = FormData()
#         for k, v in payload.items():
#             form.add_field(k, v)
#
#         # 一帧 PCM 所需字节数
#         frame_bytes = int(
#             self.opus_encoder.sample_rate
#             * self.opus_encoder.channels
#             * self.opus_encoder.frame_size_ms
#             / 1000
#             * 2
#         )
#
#         pcm_buffer = bytearray()
#         opus_datas_cache = []
#         segment_count = 0
#
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(self.api_url, data=form, timeout=30) as resp:
#                     if resp.status != 200:
#                         logger.bind(tag=TAG).error(
#                             f"TTS请求失败: {resp.status}, {await resp.text()}"
#                         )
#                         self.tts_audio_queue.put((SentenceType.LAST, [], None))
#                         return
#
#                     self.tts_audio_queue.put((SentenceType.FIRST, [], text))
#
#                     # 逐块读取音频数据
#                     async for chunk in resp.content.iter_any():
#                         if not chunk:
#                             continue
#
#                         pcm_buffer.extend(chunk)
#
#                         # 每够一帧就编码发送
#                         while len(pcm_buffer) >= frame_bytes:
#                             frame = bytes(pcm_buffer[:frame_bytes])
#                             del pcm_buffer[:frame_bytes]
#
#                             # 添加错误处理
#                             try:
#                                 opus = self.opus_encoder.encode_pcm_to_opus_stream(frame, end_of_stream=False, callback=self.handle_opus)
#                                 if opus:
#                                     if segment_count < 10:
#                                         self.tts_audio_queue.put((SentenceType.MIDDLE, opus, None))
#                                         segment_count += 1
#                                     else:
#                                         opus_datas_cache.extend(opus)
#                             except Exception as e:
#                                 logger.bind(tag=TAG).error(f"Opus编码失败: {e}")
#                                 continue
#
#                     # 处理剩余不足一帧的数据
#                     if pcm_buffer:
#                         # 确保最后一帧长度正确
#                         if len(pcm_buffer) < frame_bytes:
#                             # 填充最后一帧
#                             padding = b'\x00' * (frame_bytes - len(pcm_buffer))
#                             pcm_buffer.extend(padding)
#
#                         try:
#                             opus = self.opus_encoder.encode_pcm_to_opus_stream(bytes(pcm_buffer), end_of_stream=True, callback=self.handle_opus)
#                             if opus:
#                                 if segment_count < 10:
#                                     self.tts_audio_queue.put((SentenceType.MIDDLE, opus, None))
#                                     segment_count += 1
#                                 else:
#                                     opus_datas_cache.extend(opus)
#                         except Exception as e:
#                             logger.bind(tag=TAG).error(f"最后一帧Opus编码失败: {e}")
#
#                     # 如果不是前10个片段，最后统一发送缓存
#                     if segment_count >= 10 and opus_datas_cache:
#                         self.tts_audio_queue.put((SentenceType.MIDDLE, opus_datas_cache, None))
#
#                     # 如果是最后一段，输出音频获取完毕
#                     if is_last:
#                         self._process_before_stop_play_files()
#                         # 发送结束标记
#                         self.tts_audio_queue.put((SentenceType.LAST, [], None))
#
#         except Exception as e:
#             logger.bind(tag=TAG).error(f"TTS请求异常: {e}\n{traceback.format_exc()}")
#             self.tts_audio_queue.put((SentenceType.LAST, [], None))
#
#     def to_tts(self, text: str) -> list:
#         """非流式TTS处理，用于测试及保存音频文件的场景
#
#         Args:
#             text: 要转换的文本
#
#         Returns:
#             list: 返回opus编码后的音频数据列表
#         """
#         start_time = time.time()
#         text = MarkdownCleaner.clean_markdown(text)
#
#         # 构建语音克隆请求参数
#         payload = {
#             "text": text,
#             "reference_text": self.reference_text,
#             "reference_audio": self.reference_audio_base64,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#             "top_k": self.top_k,
#             "max_tokens": self.max_tokens,
#             "stream": False,
#             "response_format": "pcm",
#             "pitch": self.pitch,
#             "speed": self.speed
#         }
#         headers = {
#             "Content-Type": "application/x-www-form-urlencoded",
#         }
#
#         try:
#             response = requests.post(
#                 self.api_url, data=payload, headers=headers, timeout=30
#             )
#             if response.status_code != 200:
#                 logger.bind(tag=TAG).error(
#                     f"TTS请求失败: {response.status_code}, {response.text}"
#                 )
#                 return []
#
#             logger.info(f"TTS请求成功: {text}, 耗时: {time.time() - start_time}秒")
#
#             # 使用opus编码器处理PCM数据
#             opus_datas = []
#             pcm_data = response.content
#
#             # 计算每帧的字节数
#             frame_bytes = int(
#                 self.opus_encoder.sample_rate
#                 * self.opus_encoder.channels
#                 * self.opus_encoder.frame_size_ms
#                 / 1000
#                 * 2
#             )
#
#             # 分帧处理PCM数据
#             for i in range(0, len(pcm_data), frame_bytes):
#                 frame = pcm_data[i: i + frame_bytes]
#                 if len(frame) < frame_bytes:
#                     # 最后一帧可能不足，用0填充
#                     frame = frame + b"\x00" * (frame_bytes - len(frame))
#
#                 try:
#                     opus = self.opus_encoder.encode_pcm_to_opus_stream(
#                         frame, end_of_stream=(i + frame_bytes >= len(pcm_data)), callback=lambda opus: opus_datas.append(opus)
#                     )
#                     if opus:
#                         opus_datas.extend(opus)
#                 except Exception as e:
#                     logger.bind(tag=TAG).error(f"Opus编码失败: {e}")
#                     continue
#
#             return opus_datas
#
#         except Exception as e:
#             logger.bind(tag=TAG).error(f"TTS请求异常: {e}")
#             return []