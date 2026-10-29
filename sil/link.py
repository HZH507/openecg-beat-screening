"""传输链路：分包、丢包与接收端重组（用上一包末样本保持）。"""
from __future__ import annotations

import numpy as np


def transmit(signal: np.ndarray, fs: int = 360, samples_per_packet: int = 20,
             loss_rate: float = 0.01, latency_s: float = 0.02, seed: int = 42):
    """模拟设备→上位机的数据流。

    丢包位置用上一包最后一个样本保持，避免人为制造零值尖峰；
    延迟只作为缓冲事实记录，重组时按时间戳对齐，不移动信号。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    pad = (-n) % samples_per_packet
    xp = np.concatenate([x, np.zeros(pad)])
    packets = xp.reshape(-1, samples_per_packet)

    keep = rng.random(len(packets)) >= loss_rate
    out = np.empty_like(xp)
    held = float(xp[0])
    for i, (pkt, ok) in enumerate(zip(packets, keep)):
        if ok:
            out[i * samples_per_packet:(i + 1) * samples_per_packet] = pkt
            held = float(pkt[-1])
        else:
            out[i * samples_per_packet:(i + 1) * samples_per_packet] = held

    lost = int((~keep).sum())
    stats = {
        "n_packets": int(len(packets)),
        "lost_packets": lost,
        "loss_rate_set": loss_rate,
        "loss_rate_realised": round(lost / max(len(packets), 1), 4),
        "samples_per_packet": samples_per_packet,
        "packet_ms": round(1000.0 * samples_per_packet / fs, 2),
        "latency_s": latency_s,
    }
    return out[:n], stats
