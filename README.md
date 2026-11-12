# colab-qwen38-g4

在 Google Colab **G4 GPU（96 GB 显存）** 上一键跑起 **`dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4`**（262K 上下文 + MTP 投机解码 `130~176 tok/s`），自动通过本地安全网关与 Cloudflare Tunnel 反代为 OpenAI 兼容接口。

## 1. 安装并登录 `colab` CLI

推荐使用 `colab` CLI 默认的浏览器 OAuth 登录（最小权限，无需在本机落盘宽泛的 GCP `cloud-platform` 凭据）：

```bash
uv tool install google-colab-cli
colab auth login
```

<details>
<summary>无头服务器 / 自动化环境（可选使用 ADC 登录）</summary>

若在无浏览器的远程环境中运行，可通过 `gcloud` 配置 ADC（建议使用未绑定生产 GCP 项目的独立账号）：

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```
</details>

## 2. 下载脚本并运行

```bash
git clone https://github.com/mizorewww/colab-qwen38-g4.git
cd colab-qwen38-g4

# 交互式启动（推荐选择 [1] Fast Direct NVMe 隔离 Google Drive，支持自动写入 0600 权限的 pi 配置）：
python3 run_qwen38_g4.py
```

或者不走交互式向导，直接单条命令全自动跑起来（未指定 `COLAB_SGLANG_API_KEY` 时会自动在实例端生成高强度随机密钥并打印）：

```bash
colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800 run_qwen38_g4.py
```

跑完后终端会输出公网反代地址与本次会话的随机 Key：

```text
TUNNEL_BASE_URL=https://xxxx.trycloudflare.com/v1
API_KEY=sk-qwen38-...
MODEL_ID=dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
```

### 可选安全与自定义环境变量

| 环境变量 | 说明 |
| --- | --- |
| `COLAB_SGLANG_API_KEY` | 自定义 Bearer API Key（长度需 `>= 24` 且不可使用公开示例值） |
| `ROTATE_API_KEY=1` | 在已有实例重跑时强制轮换新的随机 API Key |
| `CLOUDFLARE_TUNNEL_TOKEN` | 传入自有 Cloudflare Named Tunnel Token 以绑定固定域名（替代临时 `trycloudflare.com`） |
| `CLOUDFLARE_PUBLIC_HOSTNAME` | 配合自有 Tunnel 指定输出域名（如 `llm.example.com`） |
| `ALLOW_REMOTE_MEDIA_URLS=0` | 禁用多模态远程图片 URL 拉取，仅允许 `data:` Base64 内联图片 |
| `KERNEL_CACHE_BASE_URL` | 自定义预编译算子包镜像源（仍强制执行内置 SHA-256 哈希校验） |

## 3. 内置安全防护机制

- **强制高熵随机密钥与管理面隔离**：彻底移除静态默认密钥，单命令模式亦自动生成 192-bit 随机 Bearer Key；SGLang 额外注入独立 256-bit `--admin-api-key`。
- **本地前置安全网关（`127.0.0.1:30001` -> `127.0.0.1:30000`）**：
  - SGLang 仅绑定容器回环地址 `127.0.0.1:30000`，不直接暴露给公网隧道。
  - 网关强制恒定时间 Token 比对与来源 IP 鉴权失败限流，仅白名单放行标准 OpenAI 接口（`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`），拦截所有内部管理/调试路由。
  - 自动拦截多模态请求中的 `file://` 本地路径与云内网/元数据地址（SSRF 防护），并在开启公网隧道前执行启动自检。
- **供应链完整性校验**：对预编译 `SM120` 算子包与 `cloudflared` 二进制强制执行 SHA-256 校验，使用受控路径解压（拒绝路径穿越、符号链接与特殊权限位），并默认锁定 HuggingFace 模型仓库 commit revision。

## 4. 在 `pi` 中使用

如果用交互式向导（`python3 run_qwen38_g4.py`），脚本会以 `0600` 权限自动更新 `~/.pi/agent/models.json`。直接运行：

```bash
pi --model "colab-g4/dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4" -p "你好，请简单介绍你的上下文能力"
```

如需手动配置 `~/.pi/agent/models.json`，复制粘贴以下内容（把 `baseUrl` 和 `apiKey` 换成终端输出的值）：

```json
{
  "providers": {
    "colab-g4": {
      "name": "Colab G4 SGLang",
      "baseUrl": "https://xxxx.trycloudflare.com/v1",
      "api": "openai-completions",
      "apiKey": "sk-qwen38-<replace-with-printed-key>",
      "models": [
        {
          "id": "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4",
          "name": "Qwen 3.8 Flash Next Cybersecurity NVFP4 (Colab G4)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 65536,
          "compat": {
            "supportsStore": false,
            "supportsDeveloperRole": false
          },
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh"
          },
          "samplingParams": {
            "temperature": 1.0,
            "top_p": 0.95
          },
          "samplingParamsByThinkingLevel": {
            "off": {
              "temperature": 0.7,
              "top_p": 0.8
            }
          }
        }
      ]
    }
  }
}
```

## 5. 用完关闭实例（停止计费）

```bash
colab stop -s qwen38-g4
```
