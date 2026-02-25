"""网易云音乐 API 封装 - 实现 weapi 加密协议，支持私信相关接口。"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import uuid

logger = logging.getLogger(__name__)

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# weapi 加密常量
_IV = b"0102030405060708"
_PRESET_KEY = b"0CoJUm6Qyw8W8jud"
_BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_RSA_PUBLIC_KEY_MODULUS = (
    "00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7"
    "b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280"
    "104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932"
    "575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b"
    "3ece0462db0a22b8e7"
)
_RSA_PUBLIC_KEY_EXPONENT = 0x10001

_BASE_URL = "https://music.163.com"

_COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://music.163.com/",
    "Origin": "https://music.163.com",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}


def _aes_encrypt(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, _IV)
    return base64.b64encode(cipher.encrypt(pad(data, AES.block_size)))


def _rsa_encrypt(text: str) -> str:
    reversed_text = text[::-1]
    text_int = int(reversed_text.encode().hex(), 16)
    modulus = int(_RSA_PUBLIC_KEY_MODULUS, 16)
    encrypted = pow(text_int, _RSA_PUBLIC_KEY_EXPONENT, modulus)
    return format(encrypted, "0256x")


def _weapi_encrypt(data: dict) -> dict:
    """weapi 加密：双层 AES + RSA。"""
    text = json.dumps(data).encode()

    secret_key = "".join(secrets.choice(_BASE62) for _ in range(16))

    first = _aes_encrypt(text, _PRESET_KEY)
    params = _aes_encrypt(first, secret_key.encode())
    enc_sec_key = _rsa_encrypt(secret_key)

    return {
        "params": params.decode(),
        "encSecKey": enc_sec_key,
    }


def _generate_nuid() -> str:
    return secrets.token_hex(16)


def _generate_nmtid() -> str:
    return "0" + secrets.token_hex(15) + "0"


class NeteaseAPI:
    """网易云音乐 API 客户端。"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_COMMON_HEADERS)
        self.csrf_token = ""
        self.my_uid: int | None = None
        self.my_nickname: str = ""
        self.debug = False

    def init_session(self):
        """访问主页建立真实的浏览器会话，获取初始 cookies。"""
        # 先设置一些初始 cookies，模拟真实浏览器
        nuid = _generate_nuid()
        self.session.cookies.set("_ntes_nuid", nuid, domain=".163.com")
        self.session.cookies.set(
            "_ntes_nnid", f"{nuid},{int(time.time() * 1000)}", domain=".163.com"
        )
        self.session.cookies.set("NMTID", _generate_nmtid(), domain=".163.com")

        # 访问主页获取服务端 cookies
        resp = self.session.get(
            "https://music.163.com/",
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        resp.raise_for_status()

        # 访问匿名令牌接口，建立合法 session
        self._post("/register/anonimous", {
            "username": f"guest_{secrets.token_hex(8)}",
        })

    def _post(self, path: str, data: dict) -> dict:
        """发送加密的 weapi 请求。"""
        url = f"{_BASE_URL}/weapi{path}"
        if self.csrf_token:
            data["csrf_token"] = self.csrf_token

        encrypted = _weapi_encrypt(data)
        resp = self.session.post(url, data=encrypted)
        resp.raise_for_status()
        result = resp.json()
        if self.debug:
            import sys
            print(f"[DEBUG] POST {path} -> code={result.get('code')}", file=sys.stderr)
        return result

    def _extract_csrf(self):
        """从 cookies 中提取 csrf token。"""
        for cookie in self.session.cookies:
            if cookie.name == "__csrf":
                self.csrf_token = cookie.value
                return

    # ── 登录 ──────────────────────────────────────────────

    def login_with_cookie(self, music_u: str):
        """使用 MUSIC_U cookie 直接登录。"""
        self.session.cookies.set("MUSIC_U", music_u, domain=".music.163.com")
        self._extract_csrf()

    def login_qrcode_key(self) -> str:
        """获取二维码登录的 unikey。"""
        result = self._post("/login/qrcode/unikey", {"type": 1})
        if self.debug:
            import sys
            print(f"[DEBUG] unikey response: {result}", file=sys.stderr)
        return result.get("unikey", "")

    def login_qrcode_check(self, key: str) -> tuple[int, str, dict]:
        """检查二维码扫描状态。返回 (code, message, full_result)。
        800=过期 801=等待扫描 802=已扫待确认 803=登录成功
        """
        result = self._post("/login/qrcode/client/login", {
            "key": key,
            "type": 1,
        })
        code = result.get("code", 0)
        message = result.get("message", "")
        if self.debug:
            import sys
            print(f"[DEBUG] qr check: code={code} msg={message}", file=sys.stderr)
        return code, message, result

    def login_status(self) -> dict | None:
        """获取当前登录状态。"""
        resp = self.session.post(
            f"{_BASE_URL}/api/w/nuser/account/get",
            data=_weapi_encrypt({}),
        )
        try:
            data = resp.json()
        except (ValueError, Exception):
            logger.error("login_status 响应非 JSON (status=%s)", resp.status_code)
            return None
        profile = data.get("profile")
        if profile:
            self.my_uid = profile.get("userId")
            self.my_nickname = profile.get("nickname", "")
        return profile

    # ── 私信 ──────────────────────────────────────────────

    def get_private_msg_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """获取私信联系人列表。"""
        result = self._post("/msg/private/users", {
            "offset": offset,
            "limit": limit,
            "total": "true",
        })
        return result.get("msgs", [])

    def get_private_msg_history(
        self, uid: int, limit: int = 30, before: int = 0
    ) -> list[dict]:
        """获取与某人的聊天记录。before=0 表示从最新开始。"""
        result = self._post("/msg/private/history", {
            "userId": uid,
            "limit": limit,
            "time": before,
            "total": "true",
        })
        return result.get("msgs", [])

    def fetch_all_messages(
        self, uid: int, on_progress: callable = None
    ) -> list[dict]:
        """分页拉取与某人的全部聊天记录。"""
        all_msgs: list[dict] = []
        seen_ids: set[int] = set()
        before = 0
        page_size = 100

        while True:
            msgs = self.get_private_msg_history(uid, limit=page_size, before=before)
            if not msgs:
                break

            # 去重：防止重复消息导致无限循环
            new_msgs = []
            for m in msgs:
                msg_id = m.get("id", 0)
                if msg_id and msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    new_msgs.append(m)

            if not new_msgs:
                break

            all_msgs.extend(new_msgs)
            if on_progress:
                on_progress(len(all_msgs))

            # 用本批最早消息的时间戳继续向前翻页
            before = msgs[-1].get("time", 0)

            time.sleep(0.3)

        return all_msgs


def make_qr_url(key: str) -> str:
    """构造二维码 URL。"""
    return f"https://music.163.com/login?codekey={key}"
