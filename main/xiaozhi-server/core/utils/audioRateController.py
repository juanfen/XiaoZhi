import time
import asyncio
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class AudioRateController:
    """
    音频速率控制器 - 按照60ms帧时长精确控制音频发送
    解决高并发下的时间累积误差问题

    关键改进：
    1. 单一时间基准（start_timestamp 只初始化一次）
    2. 每次检查队列时重新计算 elapsed_ms，避免累积误差
    3. 分离"检查时间"和"发送"两个操作
    4. 支持高并发而不产生延迟
    """

    def __init__(self, frame_duration=60):
        """
        Args:
            frame_duration: 单个音频帧时长（毫秒），默认60ms
        """
        self.frame_duration = frame_duration
        self.queue = []
        self.play_position = 0  # 虚拟播放位置（毫秒）
        self.start_timestamp = None  # 开始时间戳（只读，不修改）
        self.pending_send_task = None
        self.logger = logger

    def reset(self):
        """重置控制器状态"""
        if self.pending_send_task and not self.pending_send_task.done():
            self.pending_send_task.cancel()
            try:
                # 等待任务取消完成
                asyncio.get_event_loop().run_until_complete(self.pending_send_task)
            except asyncio.CancelledError:
                pass

        self.queue.clear()
        self.play_position = 0
        self.start_timestamp = time.time()

    def add_audio(self, opus_packet):
        """添加音频包到队列"""
        self.queue.append(("audio", opus_packet))

    def _get_elapsed_ms(self):
        """获取已经过的时间（毫秒）"""
        if self.start_timestamp is None:
            return 0
        return (time.time() - self.start_timestamp) * 1000

    async def check_queue(self, send_audio_callback):
        """
        检查队列并按时发送音频/消息

        Args:
            send_audio_callback: 发送音频的回调函数 async def(opus_packet)
        """
        if self.start_timestamp is None:
            self.reset()

        while self.queue:
            item = self.queue[0]
            item_type = item[0]

            if item_type == "audio":
                _, opus_packet = item

                # 计算时间差
                elapsed_ms = self._get_elapsed_ms()
                output_ms = self.play_position

                if elapsed_ms < output_ms:
                    # 还不到发送时间，计算等待时长
                    wait_ms = output_ms - elapsed_ms

                    # 等待后继续检查（允许被中断）
                    try:
                        await asyncio.sleep(wait_ms / 1000)
                    except asyncio.CancelledError:
                        self.logger.bind(tag=TAG).debug("音频发送任务被取消")
                        raise

                    # 继续循环检查（时间可能已到）
                    continue

                # 时间已到，发送音频
                self.queue.pop(0)
                self.play_position += self.frame_duration

                try:
                    await send_audio_callback(opus_packet)
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"发送音频失败: {e}")
                    raise


    async def start_sending(self, send_audio_callback, send_message_callback=None):
        """
        启动异步发送任务

        Args:
            send_audio_callback: 发送音频的回调函数
            send_message_callback: 发送消息的回调函数

        Returns:
            asyncio.Task: 发送任务
        """
        async def _send_loop():
            try:
                while True:
                    await self.check_queue(send_audio_callback, send_message_callback)
                    # 如果队列空了，短暂等待后再检查（避免 busy loop）
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                self.logger.bind(tag=TAG).info("音频发送循环已停止")
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"音频发送循环异常: {e}")

        self.pending_send_task = asyncio.create_task(_send_loop())
        return self.pending_send_task

    def stop_sending(self):
        """停止发送任务"""
        if self.pending_send_task and not self.pending_send_task.done():
            self.pending_send_task.cancel()
            self.logger.bind(tag=TAG).debug("已取消音频发送任务")
