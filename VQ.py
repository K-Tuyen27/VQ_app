import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import heapq
import io
import subprocess
import sys
import os

# HUFFMAN CODING UTILITIES

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freqs):
    heap = [HuffmanNode(sym, f) for sym, f in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)
    return heap[0]


def build_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(data):
    freqs = {}
    for symbol in data:
        freqs[symbol] = freqs.get(symbol, 0) + 1
    tree = build_huffman_tree(freqs)
    codebook = build_huffman_codes(tree)
    encoded_data = ''.join([codebook[s] for s in data])
    return encoded_data, codebook


def huffman_decode(encoded_data, codebook):
    reverse_codebook = {v: k for k, v in codebook.items()}
    code, decoded = "", []
    for bit in encoded_data:
        code += bit
        if code in reverse_codebook:
            decoded.append(reverse_codebook[code])
            code = ""
    return np.array(decoded, dtype=int)

# VECTOR QUANTIZATION (LBG)
def lbg_algorithm(vectors, M, max_iter=30, epsilon=1e-4):
    codebook = [np.mean(vectors, axis=0)]
    distortion_history = []

    while len(codebook) < M:
        new_codebook = []
        for c in codebook:
            new_codebook.append(c * (1 + epsilon))
            new_codebook.append(c * (1 - epsilon))
        codebook = new_codebook

        for iteration in range(max_iter):
            distances = np.linalg.norm(vectors[:, None] - np.array(codebook)[None, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            new_codebook = []
            for i in range(len(codebook)):
                assigned = vectors[nearest == i]
                if len(assigned) > 0:
                    new_codebook.append(np.mean(assigned, axis=0))
                else:
                    new_codebook.append(codebook[i])
            new_codebook = np.array(new_codebook)

            dist = np.mean(np.min(np.linalg.norm(vectors[:, None] - new_codebook[None, :], axis=2) ** 2, axis=1))
            distortion_history.append(dist)

            if np.allclose(new_codebook, codebook, atol=1e-3):
                break
            codebook = new_codebook

    return np.array(codebook), distortion_history

# STREAMLIT APP
def app():
    st.set_page_config(page_title="Vector Quantization Image Compressor", layout="wide")
    st.title("🧠 Vector Quantization Image Compression (LBG + Huffman)")

    st.sidebar.header("⚙️ Tùy chọn cấu hình")
    uploaded = st.sidebar.file_uploader("📤 Tải ảnh lên", type=["png", "jpg", "jpeg"])
    block_size = st.sidebar.selectbox("Kích thước khối", [2, 4, 8], index=1)
    M = st.sidebar.selectbox("Số codebook (M)", [16, 32, 64, 128, 256], index=2)
    use_huffman = st.sidebar.checkbox("Dùng Huffman Coding", value=True)
    show_compare = st.sidebar.checkbox("Hiển thị so sánh ảnh", value=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        st.image(img_np, caption="Ảnh gốc", use_column_width=True)
        H, W, C = img_np.shape
        st.write(f"Kích thước ảnh: **{W}×{H}×{C}**")

        # Chia khối
        def extract_blocks(img, bs):
            h, w, c = img.shape
            blocks = []
            for i in range(0, h, bs):
                for j in range(0, w, bs):
                    block = img[i:i+bs, j:j+bs]
                    if block.shape == (bs, bs, c):
                        blocks.append(block.flatten())
            return np.array(blocks)

        vectors = extract_blocks(img_np, block_size)
        st.write(f"Tổng số vector: {len(vectors)}")

        st.info("⏳ Đang huấn luyện codebook bằng LBG...")
        codebook, distortion_hist = lbg_algorithm(vectors, M)
        st.success("✅ Hoàn tất huấn luyện!")

        distances = np.linalg.norm(vectors[:, None] - codebook[None, :], axis=2)
        indices = np.argmin(distances, axis=1)

        if use_huffman:
            encoded_data, codebook_huff = huffman_encode(indices.tolist())
            bits_len = len(encoded_data)
            st.write(f"📦 Kích thước nén (Huffman): **{bits_len/8/1024:.2f} KB**")
        else:
            bits_len = len(indices) * np.log2(M)
            st.write(f"📦 Kích thước nén (raw index): **{bits_len/8/1024:.2f} KB**")

        if use_huffman:
            decoded_indices = huffman_decode(encoded_data, codebook_huff)
        else:
            decoded_indices = indices

        recon_blocks = codebook[decoded_indices]
        recon_img = np.zeros_like(img_np)
        idx = 0
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                if i+block_size <= H and j+block_size <= W:
                    recon_img[i:i+block_size, j:j+block_size] = recon_blocks[idx].reshape(block_size, block_size, C)
                    idx += 1

        fig, ax = plt.subplots()
        ax.plot(distortion_hist)
        ax.set_title("Giảm distortion trong quá trình LBG")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distortion")
        st.pyplot(fig)

        num_pixels = H * W * C
        bpp = bits_len / num_pixels
        compression_ratio = (8 * C) / bpp
        st.write(f"**Bits/pixel:** {bpp:.3f},  **Tỷ lệ nén:** {compression_ratio:.2f}×")

        psnr_val = psnr(img_np, recon_img, data_range=255)
        ssim_val = ssim(img_np, recon_img, channel_axis=2)
        st.write(f"**PSNR:** {psnr_val:.2f} dB,  **SSIM:** {ssim_val:.4f}")

        if show_compare:
            diff = np.abs(img_np.astype(np.int16) - recon_img.astype(np.int16))
            dist_map = np.mean(diff, axis=2)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_np, caption="Ảnh gốc", use_column_width=True)
            with col2:
                st.image(recon_img, caption="Ảnh nén (tái tạo)", use_column_width=True)
            with col3:
                fig2, ax2 = plt.subplots()
                ax2.imshow(dist_map, cmap='hot')
                ax2.set_title("Sai khác (Distortion map)")
                ax2.axis("off")
                st.pyplot(fig2)

        img_bytes = io.BytesIO()
        Image.fromarray(recon_img.astype(np.uint8)).save(img_bytes, format="PNG")
        st.download_button("📥 Tải ảnh tái tạo", data=img_bytes.getvalue(), file_name="vq_reconstructed.png", mime="image/png")

    # Nếu đang chạy Streamlit, gọi trực tiếp app
    if "streamlit" in sys.modules:
        app()
    else:
        # Chạy Streamlit bằng subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
