# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 00:11:43 2025

@author: ktrpt
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

plt.rcParams['font.family'] = 'MS Gothic'

st.set_page_config(page_title="肩関節角度レポート作成アプリ", layout="centered")
st.title("CSV-左右肩関節外転動作の視覚化")

# CSVアップロード
uploaded_file = st.file_uploader("MediaPipe出力のCSVファイルをアップロード", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSVファイルを読み込みました")
    st.dataframe(df.head())

    # データ準備
    n_frames = len(df)
    n_landmarks = 33
    pos_data_x = np.zeros((n_frames, n_landmarks))
    pos_data_y = np.zeros((n_frames, n_landmarks))
    for i in range(n_landmarks):
        pos_data_x[:, i] = df[f"x_{i}_px"].values
        pos_data_y[:, i] = -df[f"y_{i}_px"].values  # 上方向を正に

    def calangle(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.degrees(np.arccos(dot / (norm1 * norm2)))

    # 座標
    mid_head = np.array([ (pos_data_x[:, 2] + pos_data_x[:, 5]) / 2, (pos_data_y[:, 2] + pos_data_y[:, 5]) / 2 ]).T
    l_shoul = np.array([ pos_data_x[:, 11], pos_data_y[:, 11] ]).T
    l_elb   = np.array([ pos_data_x[:, 13], pos_data_y[:, 13] ]).T
    l_wri   = np.array([ pos_data_x[:, 15], pos_data_y[:, 15] ]).T
    r_shoul = np.array([ pos_data_x[:, 12], pos_data_y[:, 12] ]).T
    r_elb   = np.array([ pos_data_x[:, 14], pos_data_y[:, 14] ]).T
    r_wri   = np.array([ pos_data_x[:, 16], pos_data_y[:, 16] ]).T

    # 左肩
    vertical = np.array([0, 1])
    horizontal = np.array([1, 0])
    vec_abd_l = l_elb - l_shoul
    ang_abd_l = 180 - np.array([calangle(v, vertical) for v in vec_abd_l])
    max_abd_l = np.max(ang_abd_l)
    tmg_abd_l = np.argmax(ang_abd_l)
    vec_elev_l = r_shoul - l_shoul
    ang_elev_l = 180 - np.array([calangle(v, horizontal) for v in vec_elev_l])
    max_elev_l = np.max(ang_elev_l)

    # 右肩
    vec_abd_r = r_elb - r_shoul
    ang_abd_r = 180 - np.array([calangle(v, vertical) for v in vec_abd_r])
    max_abd_r = np.max(ang_abd_r)
    tmg_abd_r = np.argmax(ang_abd_r)
    vec_elev_r = l_shoul - r_shoul
    ang_elev_r = np.array([calangle(v, horizontal) for v in vec_elev_r])
    max_elev_r = np.max(ang_elev_r)

    def plot_stick(mid_head, l_shoul, r_shoul, elb, wri, side="L"):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.grid(alpha=0.2)

        ax.scatter(mid_head[0], mid_head[1], color='magenta', s=300, alpha=0.8)
        ax.scatter(l_shoul[0], l_shoul[1], color='gray', s=100, alpha=0.8)
        ax.scatter(r_shoul[0], r_shoul[1], color='gray', s=100, alpha=0.8)
        ax.scatter(elb[0], elb[1], color='gray', s=100, alpha=0.8)
        ax.scatter(wri[0], wri[1], color='crimson', s=100, alpha=0.8)

        ax.plot([l_shoul[0], r_shoul[0]], [l_shoul[1], r_shoul[1]], color='gray', linewidth=2, alpha=0.8)
        if side == "L":
            ax.plot([l_shoul[0], elb[0]], [l_shoul[1], elb[1]], color='gray', linewidth=2, alpha=0.8)
        else:
            ax.plot([r_shoul[0], elb[0]], [r_shoul[1], elb[1]], color='gray', linewidth=2, alpha=0.8)
        ax.plot([elb[0], wri[0]], [elb[1], wri[1]], color='gray', linewidth=2, alpha=0.8)

        return fig

    fig_l = plot_stick(mid_head[tmg_abd_l] - mid_head[0], l_shoul[tmg_abd_l] - mid_head[0],
                       r_shoul[tmg_abd_l] - mid_head[0], l_elb[tmg_abd_l] - mid_head[0], l_wri[tmg_abd_l] - mid_head[0], side="L")
    fig_r = plot_stick(mid_head[tmg_abd_r] - mid_head[0], l_shoul[tmg_abd_r] - mid_head[0],
                       r_shoul[tmg_abd_r] - mid_head[0], r_elb[tmg_abd_r] - mid_head[0], r_wri[tmg_abd_r] - mid_head[0], side="R")

    tmpdir = tempfile.gettempdir()
    path_l = os.path.join(tmpdir, "left_report.png")
    path_r = os.path.join(tmpdir, "right_report.png")
    fig_l.savefig(path_l)
    fig_r.savefig(path_r)

    st.image(path_l, caption="左肩スティックピクチャ", use_column_width=True)
    st.markdown(f"### 左肩 可動域まとめ")
    st.markdown(f"- 最大肩外転角：**{max_abd_l:.1f}°**")
    st.markdown(f"- 最大肩挙上角：**{max_elev_l:.1f}°**")

    st.image(path_r, caption="右肩スティックピクチャ", use_column_width=True)
    st.markdown(f"### 右肩 可動域まとめ")
    st.markdown(f"- 最大肩外転角：**{max_abd_r:.1f}°**")
    st.markdown(f"- 最大肩挙上角：**{max_elev_r:.1f}°**")

    with open(path_l, "rb") as f1:
        btn1 = st.download_button("左肩レポート画像をダウンロード", f1, file_name="left_report.png")
    if btn1:
        try:
            os.remove(path_l)
        except Exception as e:
            st.warning(f"左肩画像の削除に失敗しました: {e}")

    with open(path_r, "rb") as f2:
        btn2 = st.download_button("右肩レポート画像をダウンロード", f2, file_name="right_report.png")
    if btn2:
        try:
            os.remove(path_r)
        except Exception as e:
            st.warning(f"右肩画像の削除に失敗しました: {e}")
