import re
import asyncio
from datetime import datetime
from typing import Optional, List, Tuple

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image, Plain, Node, Nodes, At
from astrbot.core.platform.astr_message_event import AstrMessageEvent

# 导入模块
from .data_manager import DataManager
from .image_manager import ImageManager
from .api_manager import ApiManager
from .utils import norm_id, extract_image_urls_from_text


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "支持第三方所有OpenAI绘图格式和原生Google Gemini 终极缝合怪，文生图/图生图插件",
    "1.8.1",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config

        self.data_mgr = DataManager(StarTools.get_data_dir(), config)
        self.img_mgr = ImageManager(config)
        self.api_mgr = ApiManager(config)

    async def initialize(self):
        await self.data_mgr.initialize()
        if not self.conf.get("generic_api_keys") and not self.conf.get("gemini_api_keys"):
            logger.warning("FigurinePro: 未配置任何 API Key")
        logger.info("FigurinePro 插件已加载 (异步任务+即时反馈版 v1.8.3)")

    def is_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        """获取机器人自身的 QQ/ID，用于过滤"""
        if hasattr(event, "robot") and event.robot:
            return str(event.robot.id)
        return str(self.context.get_self_id()) if hasattr(self.context, "get_self_id") else ""

    def _save_config(self):
        try:
            if hasattr(self.conf, "save") and callable(self.conf.save):
                self.conf.save()
        except Exception as e:
            logger.warning(f"FigurinePro Config Save Failed: {e}")

    def _process_prompt_and_preset(self, prompt: str) -> Tuple[str, str]:
        sorted_keys = sorted(self.data_mgr.prompt_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in prompt:
                preset_content = self.data_mgr.prompt_map[key]
                final_prompt = f"{preset_content} , {prompt}"
                return final_prompt, key
        return prompt, "自定义"

    def _get_quota_str(self, deduction: dict, uid: str) -> str:
        if deduction["source"] == "free":
            return "∞"
        else:
            return str(self.data_mgr.get_user_count(uid))

    async def _check_quota(self, event, uid, gid, cost) -> dict:
        res = {"allowed": False, "source": None, "msg": ""}
        if self.is_admin(event) or uid in (self.conf.get("user_whitelist") or []):
            res["allowed"] = True;
            res["source"] = "free";
            return res
        if gid and gid in (self.conf.get("group_whitelist") or []):
            res["allowed"] = True;
            res["source"] = "free";
            return res
        if uid in (self.conf.get("user_blacklist") or []): return res
        if gid and gid in (self.conf.get("group_blacklist") or []): return res

        enable_u = self.conf.get("enable_user_limit", True)
        enable_g = self.conf.get("enable_group_limit", False)
        if not enable_u and not enable_g:
            res["allowed"] = True;
            res["source"] = "free";
            return res

        u_bal = self.data_mgr.get_user_count(uid)
        if enable_u and u_bal >= cost:
            res["allowed"] = True;
            res["source"] = "user";
            return res
        if gid and enable_g:
            g_bal = self.data_mgr.get_group_count(gid)
            if g_bal >= cost:
                res["allowed"] = True;
                res["source"] = "group";
                return res

        res["msg"] = f"❌ 次数不足 (需{cost}次)。用户剩余:{u_bal}"
        return res

    # ================= 核心：后台生成逻辑封装 =================

    async def _run_background_task(self, event: AstrMessageEvent, images: List[bytes],
                                   prompt: str, preset_name: str, deduction: dict, uid: str, gid: str, cost: int):
        """
        后台执行生成任务，并在完成后主动发送消息。
        """
        try:
            # 1. 扣费
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(uid, cost)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(gid, cost)

            # 2. 调用 API
            model = self.conf.get("model", "nano-banana")
            start_time = datetime.now()

            # 此处不发“开始绘制”消息了，因为前面已经发了“收到请求”

            res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)

            # 3. 处理结果
            if isinstance(res, bytes):
                elapsed = (datetime.now() - start_time).total_seconds()
                await self.data_mgr.record_usage(uid, gid)

                quota_str = self._get_quota_str(deduction, uid)
                # 构建成功文案
                info_text = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name} | 剩余: {quota_str}"
                if self.conf.get("show_model_info", False):
                    info_text += f" | {model}"

                # 4. 主动发送结果 (这是关键，LLM工具流里全靠这个发图)
                chain = event.chain_result([Image.fromBytes(res), Plain(info_text)])
                await event.send(chain)
            else:
                # 失败反馈
                await event.send(event.chain_result([Plain(f"❌ 生成失败: {res}")]))

        except Exception as e:
            logger.error(f"Background task error: {e}")
            await event.send(event.chain_result([Plain(f"❌ 系统错误: {e}")]))

    # ================= LLM 工具调用 (Tool Calling) =================

    @filter.llm_tool(name="shoubanhua_draw_image")
    async def text_to_image_tool(self, event: AstrMessageEvent, prompt: str):
        '''根据文本描述生成图片（文生图）。
        Args:
            prompt(string): 图片生成的提示词。
        '''
        # 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "LLM工具调用已禁用，请使用 #文生图 或 #bnn 指令。"
        # 1. 计算预设
        final_prompt, preset_name = self._process_prompt_and_preset(prompt)

        # 2. 【核心修改】立即发送反馈，不等待任何处理
        await event.send(event.chain_result([Plain(f"🎨 收到文生图请求，正在生成 [{preset_name}]，请稍候...")]))

        # 3. 检查配额
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        cost = 1
        deduction = await self._check_quota(event, uid, gid, cost)
        if not deduction["allowed"]:
            return deduction["msg"]

        # 4. 启动后台任务 (Fire-and-forget)
        asyncio.create_task(
            self._run_background_task(event, [], final_prompt, preset_name, deduction, uid, gid, cost)
        )

        # 5. 立刻返回给 LLM，结束对话轮次，避免超时
        return f"任务已受理，预设：{preset_name}。图片生成中，完成后将自动发送。"

    @filter.llm_tool(name="shoubanhua_edit_image")
    async def image_edit_tool(self, event: AstrMessageEvent, prompt: str, use_message_images: bool = True,
                              task_types: str = "id"):
        '''编辑用户发送的图片或引用的图片（图生图）。
        Args:
            prompt(string): 图片编辑提示词
            use_message_images(boolean): 默认 true
            task_types(string): 任务类型
        '''
        # 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "LLM工具调用已禁用，请使用 #手办化 等指令。"
        
        # 1. 计算预设
        processed_prompt, preset_name = self._process_prompt_and_preset(prompt)
        final_prompt = f"(Task Type: {task_types}) {processed_prompt}"

        # 2. 【核心修改】立即发送反馈
        await event.send(
            event.chain_result([Plain(f"🎨 收到图生图请求，正在提取图片并生成 [{preset_name}]，请耐心等待...")]))

        # 3. 提取图片 (耗时操作，但此时已发反馈，用户不会觉得卡死)
        images = []
        if use_message_images:
            bot_id = self._get_bot_id(event)
            images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id)

        if not images:
            # 如果没图，再发一条提示
            await event.send(event.chain_result([Plain("❌ 未检测到图片，请发送或引用图片。")]))
            return "失败：未检测到图片。"

        # 4. 检查配额
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        cost = 1
        deduction = await self._check_quota(event, uid, gid, cost)
        if not deduction["allowed"]:
            return deduction["msg"]

        # 5. 启动后台任务
        asyncio.create_task(
            self._run_background_task(event, images, final_prompt, preset_name, deduction, uid, gid, cost)
        )

        return f"任务已受理，预设：{preset_name}。图片生成中，完成后将自动发送。"

    # ================= 传统指令触发 =================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent, ctx=None):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text: return

        parts = text.split(maxsplit=1)
        cmd_raw = parts[0]
        match = re.search(r"[\(（](\d+)[\)）]$", cmd_raw)
        model_idx_override = int(match.group(1)) - 1 if match else None
        base_cmd = cmd_raw[:match.start()] if match else cmd_raw

        power_kw = (self.conf.get("power_model_keyword") or "").lower()
        is_power = False
        user_prompt = ""
        preset_name = "自定义"

        extra_prefix = self.conf.get("extra_prefix", "bnn")
        is_bnn = (base_cmd == extra_prefix)

        if is_bnn:
            user_prompt = parts[1] if len(parts) > 1 else ""
            user_prompt, preset_name = self._process_prompt_and_preset(user_prompt)
        else:
            preset_prompt = self.data_mgr.get_prompt(base_cmd)
            if base_cmd == "手办化帮助":
                yield self._get_help_node(event)
                return

            if not preset_prompt: return

            if power_kw and power_kw in base_cmd.lower(): is_power = True
            user_prompt = preset_prompt
            preset_name = base_cmd

            if "%" in base_cmd: user_prompt += base_cmd.split("%", 1)[1]
            if len(parts) > 1:
                if parts[1].strip().lower() == power_kw:
                    is_power = True
                else:
                    user_prompt += " " + parts[1]

        if is_power and not self.conf.get("enable_power_model", False): is_power = False

        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        cost = self.conf.get("power_model_extra_cost", 1) + 1 if is_power else 1

        deduction = await self._check_quota(event, uid, gid, cost)
        if deduction["allowed"] is False:
            yield event.chain_result([Plain(deduction["msg"])])
            return

        # 指令模式：立刻反馈
        mode_str = "增强" if is_power else ""
        yield event.chain_result([Plain(f"🎨 收到{mode_str}请求，正在生成 [{preset_name}]...")])

        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id)

        if not is_bnn and user_prompt:
            urls = extract_image_urls_from_text(user_prompt)
            for u in urls:
                if b := await self.img_mgr.load_bytes(u): images.append(b)

        if not images and not (is_bnn and user_prompt):
            yield event.chain_result([Plain("请发送图片或提供描述。")])
            return

        model = self.conf.get("power_model_id") if is_power else self.conf.get("model", "nano-banana")
        if model_idx_override is not None and not is_power:
            all_models = [m if isinstance(m, str) else m["id"] for m in self.conf.get("model_list", [])]
            if 0 <= model_idx_override < len(all_models):
                model = all_models[model_idx_override]

        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, cost)

        start = datetime.now()
        res = await self.api_mgr.call_api(images, user_prompt, model, is_power, self.img_mgr.proxy)

        if isinstance(res, bytes):
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, gid)
            if not is_bnn: await self.data_mgr.save_preset_image(base_cmd, res)

            quota_str = self._get_quota_str(deduction, uid)
            info = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name} | 剩余: {quota_str}"
            if self.conf.get("show_model_info", False):
                info += f" | {model}"

            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"❌ 失败: {res}")])
        event.stop_event()

    @filter.command("文生图", prefix_optional=True)
    async def on_txt2img(self, event: AstrMessageEvent, ctx=None):
        raw = event.message_str.strip()
        cmd_name = "文生图"
        prompt = raw.replace(cmd_name, "").strip()
        if not prompt: yield event.chain_result([Plain("请输入描述。")]); return

        uid = norm_id(event.get_sender_id())
        deduction = await self._check_quota(event, uid, event.get_group_id(), 1)
        if not deduction["allowed"]: yield event.chain_result([Plain(deduction["msg"])]); return

        final_prompt, preset_name = self._process_prompt_and_preset(prompt)
        yield event.chain_result([Plain(f"🎨 收到请求，正在生成 [{preset_name}]...")])

        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(event.get_group_id(), 1)

        model = self.conf.get("model", "nano-banana")
        start = datetime.now()
        res = await self.api_mgr.call_api([], final_prompt, model, False, self.img_mgr.proxy)

        if isinstance(res, bytes):
            elapsed = (datetime.now() - start).total_seconds()
            quota_str = self._get_quota_str(deduction, uid)
            info = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name} | 剩余: {quota_str}"
            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"❌ {res}")])

    # 辅助方法
    def _get_help_node(self, event):
        txt = self.conf.get("help_text", "帮助文档未配置")
        bot_id = self._get_bot_id(event) or "2854196310"
        return event.chain_result([Nodes(nodes=[Node(name="手办化助手", uin=bot_id, content=[Plain(txt)])])])

    # 省略 Admin指令，它们和上一版完全一致，请确保不要覆盖掉下面的代码（lm列表, lm添加, 增加次数等）

    @filter.command("lm列表", aliases={"lmlist"}, prefix_optional=True)
    async def on_preset_list(self, event: AstrMessageEvent, ctx=None):
        presets = []
        for k, v in self.data_mgr.prompt_map.items():
            presets.append((k, v == "[内置预设]"))
        presets.sort(key=lambda x: x[0])
        if not presets: yield event.chain_result([Plain("暂无预设")]); return
        img_data = await self.img_mgr.create_preset_table(presets, self.data_mgr)
        yield event.chain_result([Image.fromBytes(img_data)])

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def on_add_preset(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        msg = event.message_str.replace("lm添加", "").replace("lma", "").strip()
        if ":" not in msg: yield event.chain_result([Plain("格式: 词:提示词")]); return
        k, v = msg.split(":", 1)
        plist = self.conf.get("prompt_list", [])
        plist = [p for p in plist if not p.startswith(k.strip() + ":")]
        plist.append(f"{k.strip()}:{v.strip()}")
        self.conf["prompt_list"] = plist
        self._save_config()
        self.data_mgr.reload_prompts()
        yield event.chain_result([Plain(f"✅ 已添加预设: {k.strip()}")])

    @filter.command("lm查看", aliases={"lmv", "lm预览"}, prefix_optional=True)
    async def on_view_preset(self, event: AstrMessageEvent, ctx=None):
        parts = event.message_str.split()
        if len(parts) < 2: yield event.chain_result([Plain("用法: #lm查看 <关键词>")]); return
        kw = parts[1].strip()
        prompt = self.data_mgr.get_prompt(kw)
        msg = f"🔍 [{kw}]:\n{prompt}" if prompt else f"❌ 未找到 [{kw}]"
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent, ctx=None):
        if not self.conf.get("enable_checkin", False): yield event.chain_result([Plain("未开启签到")]); return
        uid = norm_id(event.get_sender_id())
        msg = await self.data_mgr.process_checkin(uid)
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_count(self, event: AstrMessageEvent, ctx=None):
        uid = norm_id(event.get_sender_id())
        if self.is_admin(event):
            for seg in event.message_obj.message:
                if isinstance(seg, At): uid = str(seg.qq); break
        u_cnt = self.data_mgr.get_user_count(uid)
        msg = f"👤 用户 {uid} 剩余: {u_cnt}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self.data_mgr.get_group_count(norm_id(gid))}"
        yield event.chain_result([Plain(msg)])

    @filter.command("切换API模式", prefix_optional=True)
    async def on_switch_mode(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        mode = event.message_str.split()[-1]
        if mode in ["generic", "gemini_official"]:
            self.conf["api_mode"] = mode;
            self._save_config()
            yield event.chain_result([Plain(f"✅ 已切换为 {mode}")])
        else:
            yield event.chain_result([Plain("模式无效 (generic / gemini_official)")])

    @filter.command("切换模型", prefix_optional=True)
    async def on_switch_model(self, event: AstrMessageEvent, ctx=None):
        all_m = [m if isinstance(m, str) else m["id"] for m in self.conf.get("model_list", [])]
        parts = event.message_str.split()
        if len(parts) == 1:
            curr = self.conf.get("model", "nano-banana")
            msg = "📋 可用模型:\n" + "\n".join([f"{i + 1}. {m} {'✅' if m == curr else ''}" for i, m in enumerate(all_m)])
            yield event.chain_result([Plain(msg)]);
            return

        if not self.is_admin(event): return
        if not parts[1].isdigit(): yield event.chain_result([Plain("请输入序号")]); return
        idx = int(parts[1]) - 1
        if 0 <= idx < len(all_m):
            self.conf["model"] = all_m[idx];
            self._save_config()
            yield event.chain_result([Plain(f"✅ 切换为: {all_m[idx]}")])

    @filter.command("手办化今日统计", prefix_optional=True)
    async def on_daily_stats(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        stats = self.data_mgr.daily_stats
        today = datetime.now().strftime("%Y-%m-%d")
        if stats.get("date") != today: yield event.chain_result([Plain(f"📊 {today} 无数据")]); return

        u_top = sorted(stats["users"].items(), key=lambda x: x[1], reverse=True)[:10]
        g_top = sorted(stats["groups"].items(), key=lambda x: x[1], reverse=True)[:10]
        msg = f"📊 {today} 统计:\n👥 群排行:\n" + ("\n".join([f"{k}: {v}" for k, v in g_top]) or "无")
        msg += "\n\n👤 用户排行:\n" + ("\n".join([f"{k}: {v}" for k, v in u_top]) or "无")
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        target = None
        for seg in event.message_obj.message:
            if isinstance(seg, At): target = str(seg.qq); break

        parts = event.message_str.split()
        count = 0
        if target:
            for p in parts:
                if p.isdigit(): count = int(p)
        else:
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                target = parts[1];
                count = int(parts[2])

        if target and count:
            await self.data_mgr.add_user_count(target, count)
            yield event.chain_result([Plain(f"✅ 用户 {target} +{count}")])

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            await self.data_mgr.add_group_count(parts[1], int(parts[2]))
            yield event.chain_result([Plain(f"✅ 群 {parts[1]} +{parts[2]}")])

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2: return

        is_power = parts[1].lower() in ["p", "power", "强力"]
        keys = parts[2:] if is_power else parts[1:]

        mode = self.conf.get("api_mode", "generic")
        field = f"{'power_' if is_power else ''}{mode if mode == 'generic' else 'gemini'}_api_keys"
        if mode == "gemini_official":
            field = f"{'power_' if is_power else ''}gemini_api_keys"
        else:
            field = f"{'power_' if is_power else ''}generic_api_keys"

        curr_keys = self.conf.get(field, [])
        curr_keys.extend(keys)
        self.conf[field] = curr_keys;
        self._save_config()
        yield event.chain_result([Plain(f"✅ 已向 {field} 添加 {len(keys)} 个 Key")])

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        mode = self.conf.get("api_mode", "generic")
        base = "gemini" if mode == "gemini_official" else "generic"

        nk = self.conf.get(f"{base}_api_keys", [])
        pk = self.conf.get(f"power_{base}_api_keys", [])

        msg = f"🔑 模式: {mode}\n📌 普通池 ({len(nk)}):\n" + "\n".join([f"{k[:8]}..." for k in nk])
        msg += f"\n\n⚡ 强力池 ({len(pk)}):\n" + "\n".join([f"{k[:8]}..." for k in pk])
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2: yield event.chain_result([Plain("用法: #删除key [p] <all/序号>")]); return

        is_power = parts[1].lower() in ["p", "power"]
        idx_str = parts[2] if is_power else parts[1]

        mode = self.conf.get("api_mode", "generic")
        base = "gemini" if mode == "gemini_official" else "generic"
        field = f"{'power_' if is_power else ''}{base}_api_keys"

        if idx_str == "all":
            self.conf[field] = [];
            self._save_config()
            yield event.chain_result([Plain("✅ 已清空")])
        elif idx_str.isdigit():
            keys = self.conf.get(field, [])
            idx = int(idx_str) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx);
                self.conf[field] = keys;
                self._save_config()
                yield event.chain_result([Plain("✅ 已删除")])

    @filter.command("预设图片清理", prefix_optional=True)
    async def on_cleanup_presets(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
        count = await self.data_mgr.cleanup_old_presets(days)
        yield event.chain_result([Plain(f"✅ 清理了 {count} 张超过 {days} 天的图片")])

    @filter.command("预设图片统计", prefix_optional=True)
    async def on_preset_stats(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        cnt, size = self.data_mgr.get_preset_stats()
        yield event.chain_result([Plain(f"📊 缓存统计:\n数量: {cnt} 张\n占用: {size:.2f} MB")])

    @filter.command("手办化帮助", aliases={"lmh", "lm帮助"}, prefix_optional=True)
    async def on_help(self, event: AstrMessageEvent, ctx=None):
        yield self._get_help_node(event)