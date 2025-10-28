import streamlit as st
import random, math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Function Definitions (포스터 생성 로직) ---

def blob(center=(0.5,0.5), r=0.3, points=200, wobble=0.15, shape="circle", concavity=0):
    """
    울퉁불퉁한 도형의 좌표를 생성합니다. (원, 사각형, 다이아몬드 지원)
    concavity 매개변수는 중앙이 안으로 파고드는 정도를 조절합니다.
    """
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    # 원형 기반 반지름 계산 (wobble 및 concavity 적용)
    radii  = r * (1 + wobble*(np.random.rand(points)-0.5) - concavity * np.sin(angles * 2)**2) 

    if shape == "square":
        # 사각형 모양 계산
        half_side = r * math.sqrt(2)
        points_per_side = points // 4
        x_square = []
        y_square = []

        x_square.extend(np.linspace(center[0] - half_side/2, center[0] + half_side/2, points_per_side))
        y_square.extend([center[1] + half_side/2] * points_per_side)

        x_square.extend([center[0] + half_side/2] * points_per_side)
        y_square.extend(np.linspace(center[1] + half_side/2, center[1] - half_side/2, points_per_side))

        x_square.extend(np.linspace(center[0] + half_side/2, center[0] - half_side/2, points_per_side))
        y_square.extend([center[1] - half_side/2] * points_per_side)

        x_square.extend([center[0] - half_side/2] * points_per_side)
        y_square.extend(np.linspace(center[1] - half_side/2, center[1] + half_side/2, points_per_side))

        # Add wobble and concavity to square points
        x = np.array(x_square) + wobble * r * (np.random.rand(points)-0.5)
        y = np.array(y_square) + wobble * r * (np.random.rand(points)-0.5)
        
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        angles_square = np.arctan2(y - center[1], x - center[0])
        radii_square = dist_from_center - concavity * r * np.sin(angles_square * 2)**2
        x = center[0] + radii_square * np.cos(angles_square)
        y = center[1] + radii_square * np.sin(angles_square)


    elif shape == "diamond":
        # 다이아몬드 (마름모) 모양 계산
        half_diag = r * math.sqrt(2) / 2
        points_per_side = points // 4
        x_diamond = []
        y_diamond = []

        x_diamond.extend(np.linspace(center[0], center[0] + half_diag, points_per_side))
        y_diamond.extend(np.linspace(center[1] + half_diag, center[1], points_per_side))

        x_diamond.extend(np.linspace(center[0] + half_diag, center[0], points_per_side))
        y_diamond.extend(np.linspace(center[1], center[1] - half_diag, points_per_side))

        x_diamond.extend(np.linspace(center[0], center[0] - half_diag, points_per_side))
        y_diamond.extend(np.linspace(center[1] - half_diag, center[1], points_per_side))

        x_diamond.extend(np.linspace(center[0] - half_diag, center[0], points_per_side))
        y_diamond.extend(np.linspace(center[1], center[1] + half_diag, points_per_side))

        # Add wobble and concavity to diamond points
        x = np.array(x_diamond) + wobble * r * (np.random.rand(points)-0.5)
        y = np.array(y_diamond) + wobble * r * (np.random.rand(points)-0.5)
        
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        angles_diamond = np.arctan2(y - center[1], x - center[0])
        radii_diamond = dist_from_center - concavity * r * np.sin(angles_diamond * 2)**2
        x = center[0] + radii_diamond * np.cos(angles_diamond)
        y = center[1] + radii_diamond * np.sin(angles_diamond)


    else: # circle shape (원형)
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
    return x, y

def generate_3d_poster(n_layers=6, seed=0, shape="circle", concavity=0):
    """
    3D 깊이감을 표현하는 추상 포스터를 생성하고 Matplotlib figure를 반환합니다.
    """
    random.seed(seed); np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(7,7), constrained_layout=True)
    ax.axis('off')
    ax.set_facecolor((0.95,0.95,0.95)) # 밝은 배경색

    for depth in range(n_layers):
        # 위치를 중앙에 몰리도록 정규 분포 사용
        cx = np.random.normal(0.5, 0.1) 
        cy = np.random.normal(0.5, 0.1) 
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))

        rr = random.uniform(0.15, 0.45)

        # 1. 그림자 (약간 이동, 어둡고 반투명)
        x_shadow, y_shadow = blob((cx,cy), r=rr, wobble=0.12, shape=shape, concavity=concavity)
        ax.fill(x_shadow+0.02, y_shadow-0.02, color=(0,0,0), alpha=0.2)

        # 2. 메인 도형 (깊이에 따라 투명도 변화)
        x_main, y_main = blob((cx,cy), r=rr, wobble=0.12, shape=shape, concavity=concavity)
        color = (random.random(), random.random(), random.random())
        alpha = 0.4 + depth*0.08  # 깊이가 깊을수록 (레이어 번호가 클수록) 투명도 증가 (더 앞에 있는 것처럼 보이도록)
        ax.fill(x_main, y_main, color=color, alpha=min(alpha,1.0))

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"3D-like Generative Poster ({shape.capitalize()})", fontsize=14, weight="bold")
    ax.text(0.05, 0.05, f"Concavity: {concavity:.2f}, Layers: {n_layers}, Seed: {seed}", 
            transform=ax.transAxes, fontsize=10, color='gray')
    
    return fig

# --- 2. Streamlit Application Logic (UI) ---

def regenerate_seed():
    """새로운 난수 시드를 생성하여 session_state에 저장합니다."""
    st.session_state.seed = random.randint(0, 99999)

# 난수 시드 초기화 (앱 실행 시 또는 새로고침 시)
if 'seed' not in st.session_state:
    regenerate_seed()


st.set_page_config(layout="wide", page_title="3D Generative Poster")

st.title("Generative Poster: 3D Layering Effect")
st.markdown("---")

# --- UI Controls in Sidebar ---
with st.sidebar:
    st.header("포스터 설정")

    # 1. Shape Selection
    selected_shape = st.selectbox(
        "도형 모양 (Shape)",
        ("circle", "square", "diamond"),
        key="shape"
    )

    # 2. Concavity Slider
    selected_concavity = st.slider(
        "오목함 (Concavity)",
        min_value=0.0,
        max_value=0.3,
        value=0.15,
        step=0.01,
        key="concavity",
        help="도형의 중앙이 안으로 파고드는 정도를 조절합니다."
    )

    # 3. Layer Count
    layer_count = st.slider(
        "레이어 수 (Depth)",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
        key="n_layers",
        help="생성할 도형의 개수를 조절하여 깊이감을 설정합니다."
    )

    st.markdown("---")
    st.subheader("새 포스터 생성")

    # 4. Regenerate Button
    st.button(
        "🎨 새 포스터 생성",
        on_click=regenerate_seed,
        type="primary"
    )

    # 5. Current Seed Display 
    st.info(f"현재 시드: **{st.session_state.seed}**")

# --- Main Plotting Area ---

# 현재 시드와 사이드바 설정값을 사용하여 포스터 생성
poster_fig = generate_3d_poster(
    n_layers=layer_count,
    seed=st.session_state.seed,
    shape=selected_shape,
    concavity=selected_concavity
)

# 생성된 포스터 그림을 Streamlit에 표시
st.pyplot(poster_fig)
st.caption("그림은 Matplotlib을 사용하여 생성되었으며, 그림자, 투명도, 레이어링을 통해 3차원적 깊이감을 연출합니다.")
