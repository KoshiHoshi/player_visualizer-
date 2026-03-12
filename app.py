import streamlit as st
import pandas as pd
import os
import json
from PIL import Image
import plotly.graph_objects as go
import time
import base64
from io import BytesIO
import numpy as np

st.set_page_config(layout="wide", page_title="LILA BLACK - Player Visualizer")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MINIMAP_DIR = os.path.join(BASE_DIR, "minimaps")

MINIMAP_PATHS = {
    "AmbroseValley": os.path.join(MINIMAP_DIR, "AmbroseValley_Minimap.png"),
    "GrandRift":     os.path.join(MINIMAP_DIR, "GrandRift_Minimap.png"),
    "Lockdown":      os.path.join(MINIMAP_DIR, "Lockdown_Minimap.jpg"),
}

map_config = {
    "AmbroseValley": {"scale": 900,  "origin_x": -370, "origin_z": -473},
    "GrandRift":     {"scale": 581,  "origin_x": -290, "origin_z": -290},
    "Lockdown":      {"scale": 1000, "origin_x": -500, "origin_z": -500},
}

DAY_FOLDERS = ["February_10", "February_11", "February_12", "February_13", "February_14"]

HUMAN_COLORS = [
    "#00FFFF","#FF6B6B","#FFE66D","#A8E6CF",
    "#FF8B94","#B8F2E6","#FFA07A","#98D8C8",
    "#F7DC6F","#82E0AA","#F1948A","#85C1E9",
]

EVENT_STYLES = {
    "Kill":          {"color": "lime",   "symbol": "star",             "size": 14},
    "Killed":        {"color": "red",    "symbol": "x",                "size": 14},
    "BotKill":       {"color": "orange", "symbol": "star-triangle-up", "size": 12},
    "BotKilled":     {"color": "salmon", "symbol": "x-thin",           "size": 12},
    "KilledByStorm": {"color": "#AA00FF","symbol": "hexagram",         "size": 14},
    "Loot":          {"color": "yellow", "symbol": "diamond",          "size": 10},
}

def world_to_minimap(x, z, map_id):
    cfg = map_config[map_id]
    u = (x - cfg["origin_x"]) / cfg["scale"]
    v = (z - cfg["origin_z"]) / cfg["scale"]
    return u * 1024, v * 1024

def format_time(ms):
    seconds = int(ms / 1000)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def add_px_py(df, map_id):
    df = df.copy()
    coords = df.apply(lambda r: pd.Series(world_to_minimap(r["x"], r["z"], map_id)), axis=1)
    df["px"] = coords[0]
    df["py"] = coords[1]
    return df

def make_dead_zone_overlay(position_df, grid=32):
    if position_df.empty:
        return None
    bins = np.linspace(0, 1024, grid + 1)
    H, xedges, yedges = np.histogram2d(position_df["px"], position_df["py"], bins=bins)
    max_val = H.max() if H.max() > 0 else 1
    dead = np.where(H < max_val * 0.03, 1.0, 0.0)
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    xs, ys = [], []
    for i in range(grid):
        for j in range(grid):
            if dead[i, j] > 0:
                xs.append(x_centers[i])
                ys.append(y_centers[j])
    if not xs:
        return None
    cell_size = 1024 / grid
    return go.Scatter(
        x=xs, y=ys, mode="markers", name="Dead Zones",
        hovertext="Dead Zone - no players visit here", hoverinfo="text",
        marker=dict(size=cell_size * 0.9, color="rgba(0,100,255,0.25)",
                    symbol="square", line=dict(width=0)),
        showlegend=True,
    )

def make_base_fig(selected_map):
    map_img = Image.open(MINIMAP_PATHS[selected_map])
    map_b64 = pil_to_base64(map_img)
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=map_b64, x=0, y=1024,
        sizex=1024, sizey=1024,
        xref="x", yref="y",
        layer="below", sizing="stretch",
    ))
    return fig

@st.cache_data
def load_json_data():
    with open(os.path.join(OUTPUT_DIR, "events.json")) as f:
        events = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "heatmap.json")) as f:
        heatmap = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "matches.json")) as f:
        matches = json.load(f)
    events_df  = pd.DataFrame(events)
    heatmap_df = pd.DataFrame(heatmap)
    matches_df = pd.DataFrame(matches)
    return events_df, heatmap_df, matches_df

@st.cache_data
def get_match_detail(match_id, events_df):
    match_df = events_df[events_df["match_id"] == match_id].copy()
    if match_df.empty:
        return match_df
    match_df = match_df.sort_values("ts").reset_index(drop=True)
    match_start = match_df["ts"].min()
    match_df["ts_ms"] = (match_df["ts"] - match_start).astype(int)
    return match_df

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "playing"    not in st.session_state: st.session_state.playing    = False
if "cursor"     not in st.session_state: st.session_state.cursor     = 999_999_999
if "last_match" not in st.session_state: st.session_state.last_match = None

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#00FFFF; font-family:monospace; margin-bottom:0'>
    LILA BLACK — Level Design Analytics
    </h1>
    <p style='color:#888; margin-top:4px'>Player telemetry visualization tool for the Level Design team</p>
    <hr style='border-color:#333'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data..."):
    events_df, heatmap_df, matches_df = load_json_data()

# ─────────────────────────────────────────────
# SIDEBAR — top level filters
# ─────────────────────────────────────────────
st.sidebar.markdown("## Filters")

available_maps = sorted(events_df["map"].unique())
selected_map   = st.sidebar.selectbox("Select Map", available_maps)

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("Analysis Mode", ["Single Match", "Aggregate (All Matches)"])

map_events  = events_df[events_df["map"] == selected_map].copy()
map_heatmap = heatmap_df[heatmap_df["map"] == selected_map].copy()
map_matches = matches_df[matches_df["map"] == selected_map].copy()

selected_days = st.sidebar.multiselect(
    "Filter by Date", options=DAY_FOLDERS, default=DAY_FOLDERS,
    format_func=lambda x: x.replace("_", " ")
)
if selected_days:
    map_events  = map_events[map_events["date"].isin(selected_days)]
    map_heatmap = map_heatmap[map_heatmap["date"].isin(selected_days)]
    map_matches = map_matches[map_matches["date"].isin(selected_days)]

# ─────────────────────────────────────────────
# AGGREGATE MODE
# ─────────────────────────────────────────────
if view_mode == "Aggregate (All Matches)":

    st.markdown("## Aggregate Analysis — " + selected_map)
    st.caption(
        f"{map_matches['match_id'].nunique()} matches · "
        f"{map_events[map_events['human']==True]['user_id'].nunique()} unique players"
    )

    map_events_px  = add_px_py(map_events, selected_map)
    map_heatmap_px = add_px_py(map_heatmap, selected_map)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Layers")
    agg_show_traffic   = st.sidebar.checkbox("Traffic Heatmap",     value=True)
    agg_show_kills     = st.sidebar.checkbox("Kill Heatmap",        value=False)
    agg_show_deaths    = st.sidebar.checkbox("Death Heatmap",       value=False)
    agg_show_storm     = st.sidebar.checkbox("Storm Death Heatmap", value=False)
    agg_show_loot      = st.sidebar.checkbox("Loot Heatmap",        value=False)
    agg_show_deadzones = st.sidebar.checkbox("Dead Zones",          value=False)
    agg_show_markers   = st.sidebar.checkbox("Individual Markers",  value=False)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Matches",        map_matches["match_id"].nunique())
    c2.metric("Unique Players", map_events[map_events["human"]==True]["user_id"].nunique())
    c3.metric("Total Kills",    len(map_events[map_events["event"].isin(["Kill","BotKill"])]))
    c4.metric("Total Deaths",   len(map_events[map_events["event"].isin(["Killed","BotKilled"])]))
    c5.metric("Storm Deaths",   len(map_events[map_events["event"] == "KilledByStorm"]))
    c6.metric("Loot Events",    len(map_events[map_events["event"] == "Loot"]))

    with st.expander("Auto Insights", expanded=True):
        ic1, ic2, ic3 = st.columns(3)

        kill_df = map_events_px[map_events_px["event"].isin(["Kill","BotKill"])].copy()
        if not kill_df.empty:
            kill_df["grid_x"] = (kill_df["px"] // 128).astype(int)
            kill_df["grid_y"] = (kill_df["py"] // 128).astype(int)
            kill_pct = kill_df.groupby(["grid_x","grid_y"]).size().max() / len(kill_df) * 100
            ic1.markdown(f"""
**Combat Hotspot**

~{kill_pct:.0f}% of kills occur in one map sector.
Consider balancing loot or cover in this zone.
""")

        storm_df = map_events_px[map_events_px["event"] == "KilledByStorm"].copy()
        if not storm_df.empty:
            storm_df["grid_x"] = (storm_df["px"] // 128).astype(int)
            storm_df["grid_y"] = (storm_df["py"] // 128).astype(int)
            storm_pct = storm_df.groupby(["grid_x","grid_y"]).size().max() / max(len(storm_df),1) * 100
            ic2.markdown(f"""
**Storm Pressure**

~{storm_pct:.0f}% of storm deaths cluster in one zone.
The storm may be moving too fast in that area.
""")

        loot_df = map_events_px[map_events_px["event"] == "Loot"].copy()
        if not loot_df.empty:
            loot_df["grid_x"] = (loot_df["px"] // 256).astype(int)
            loot_df["grid_y"] = (loot_df["py"] // 256).astype(int)
            ignored_pct = (16 - loot_df.groupby(["grid_x","grid_y"]).ngroups) / 16 * 100
            ic3.markdown(f"""
**Loot Behavior**

~{ignored_pct:.0f}% of map sectors have zero loot activity.
Some areas may need better loot incentives.
""")

    fig = make_base_fig(selected_map)

    def add_heatmap(df_sub, colorscale, name):
        if len(df_sub) < 2: return
        fig.add_trace(go.Histogram2dContour(
            x=df_sub["px"], y=df_sub["py"],
            colorscale=colorscale, showscale=False,
            opacity=0.6, contours=dict(showlines=False),
            ncontours=25, name=name, hoverinfo="skip",
        ))

    if agg_show_traffic:
        add_heatmap(map_heatmap_px[map_heatmap_px["human"]==True], "Blues", "Traffic")
    if agg_show_kills:
        add_heatmap(map_events_px[map_events_px["event"].isin(["Kill","BotKill"])], "Reds", "Kills")
    if agg_show_deaths:
        add_heatmap(map_events_px[map_events_px["event"].isin(["Killed","BotKilled"])], "Oranges", "Deaths")
    if agg_show_storm:
        add_heatmap(map_events_px[map_events_px["event"] == "KilledByStorm"], "Purples", "Storm Deaths")
    if agg_show_loot:
        add_heatmap(map_events_px[map_events_px["event"] == "Loot"], "Greens", "Loot")

    if agg_show_deadzones:
        dz = make_dead_zone_overlay(map_heatmap_px[map_heatmap_px["human"]==True])
        if dz: fig.add_trace(dz)

    if agg_show_markers:
        for event_type, style in EVENT_STYLES.items():
            edf = map_events_px[map_events_px["event"] == event_type]
            if len(edf) > 200: edf = edf.sample(200, random_state=42)
            if edf.empty: continue
            fig.add_trace(go.Scatter(
                x=edf["px"], y=edf["py"], mode="markers", name=event_type,
                marker=dict(size=style["size"]-2, color=style["color"],
                            symbol=style["symbol"], opacity=0.7,
                            line=dict(width=1, color="white")),
                hoverinfo="skip",
            ))

    fig.update_xaxes(range=[0,1024], showgrid=False, visible=False, constrain="domain")
    fig.update_yaxes(range=[0,1024], showgrid=False, visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        legend=dict(bgcolor="rgba(0,0,0,0.7)", font=dict(color="white", size=11),
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1, x=1.01, y=1),
        height=780,
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# SINGLE MATCH MODE
# ─────────────────────────────────────────────
else:
    available_matches = map_matches.sort_values("match_id")["match_id"].unique()
    match_labels = {
        f"Match {i+1}  ({m[:8]}...)": m
        for i, m in enumerate(available_matches)
    }
    selected_label = st.sidebar.selectbox("Select Match", list(match_labels.keys()))
    selected_match = match_labels[selected_label]

    match_df = get_match_detail(selected_match, map_events)

    if match_df.empty:
        st.error("No data for this match.")
        st.stop()

    match_duration = int(match_df["ts_ms"].max())

    if st.session_state.last_match != selected_match:
        st.session_state.cursor     = match_duration
        st.session_state.playing    = False
        st.session_state.last_match = selected_match

    st.session_state.cursor = min(st.session_state.cursor, match_duration)

    # ── SIDEBAR LAYERS ────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layers")
    show_human_paths = st.sidebar.checkbox("Human Movement Paths", value=True)
    show_bot_paths   = st.sidebar.checkbox("Bot Movement Paths",   value=False)
    show_kills       = st.sidebar.checkbox("Kill Markers",         value=True)
    show_loot        = st.sidebar.checkbox("Loot Markers",         value=True)
    show_storm       = st.sidebar.checkbox("Storm Death Markers",  value=True)
    show_heatmap     = st.sidebar.checkbox("Heatmap Overlay",      value=False)
    show_deadzones   = st.sidebar.checkbox("Dead Zones",           value=False)
    heatmap_type     = st.sidebar.radio(
        "Heatmap Type", ["Kills Only","Deaths Only","Storm Deaths","Loot"]
    )
    path_opacity = st.sidebar.slider("Path Opacity", 0.1, 1.0, 0.6, 0.05)

    # ── SIDEBAR TIMELINE ──────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Timeline Playback")
    st.sidebar.caption("Showing full match by default — Reset then Play to watch!")

    sb1, sb2, sb3 = st.sidebar.columns(3)
    if sb1.button("▶ Play"):
        st.session_state.playing = True
    if sb2.button("⏸ Pause"):
        st.session_state.playing = False
    if sb3.button("⏮ Reset"):
        st.session_state.playing = False
        st.session_state.cursor  = 0

    def on_slider_change():
        st.session_state.playing = False
        st.session_state.cursor  = st.session_state.manual_slider

    st.sidebar.slider(
        "Match Time",
        min_value=0,
        max_value=max(match_duration, 1),
        value=st.session_state.cursor,
        key="manual_slider",
        on_change=on_slider_change,
    )

    ts_cursor = st.session_state.cursor
    st.sidebar.caption(f"⏱ {format_time(ts_cursor)} / {format_time(match_duration)}")

    # ── MAIN AREA ─────────────────────────────
    st.markdown("## Match Overview")
    match_info = map_matches[map_matches["match_id"] == selected_match]
    if not match_info.empty:
        match_info = match_info.iloc[0]
        num_humans = int(match_info["players"])
        num_bots   = int(match_info["bots"])
    else:
        num_humans = match_df[match_df["human"]==True]["user_id"].nunique()
        num_bots   = match_df[match_df["human"]==False]["user_id"].nunique()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Human Players", num_humans)
    c2.metric("Bots",          num_bots)
    c3.metric("Duration",      format_time(match_duration))
    c4.metric("Total Kills",   len(match_df[match_df["event"].isin(["Kill","BotKill"])]))
    c5.metric("Storm Deaths",  len(match_df[match_df["event"] == "KilledByStorm"]))
    c6.metric("Loot Events",   len(match_df[match_df["event"] == "Loot"]))

    # Live stats
    visible_df = match_df[match_df["ts_ms"] <= ts_cursor].copy()
    visible_df = add_px_py(visible_df, selected_map)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Events Shown", len(visible_df))
    c2.metric("Kills So Far", len(visible_df[visible_df["event"].isin(["Kill","BotKill"])]))
    c3.metric("Storm Deaths", len(visible_df[visible_df["event"] == "KilledByStorm"]))
    c4.metric("Loot So Far",  len(visible_df[visible_df["event"] == "Loot"]))

    # ── FIGURE ────────────────────────────────
    fig = make_base_fig(selected_map)

    if show_heatmap:
        hmap_events = {
            "Kills Only":   ["Kill","BotKill"],
            "Deaths Only":  ["Killed","BotKilled"],
            "Storm Deaths": ["KilledByStorm"],
            "Loot":         ["Loot"],
        }
        hmap_colors = {
            "Kills Only":   "Reds",
            "Deaths Only":  "Oranges",
            "Storm Deaths": "Purples",
            "Loot":         "Greens",
        }
        heat_df = visible_df[visible_df["event"].isin(hmap_events[heatmap_type])]
        if len(heat_df) > 1:
            fig.add_trace(go.Histogram2dContour(
                x=heat_df["px"], y=heat_df["py"],
                colorscale=hmap_colors[heatmap_type],
                showscale=False, opacity=0.55,
                contours=dict(showlines=False),
                ncontours=20, hoverinfo="skip",
            ))

    if show_deadzones:
        match_heat = map_heatmap[map_heatmap["match"] == selected_match].copy()
        if not match_heat.empty:
            match_heat = add_px_py(match_heat, selected_map)
            dz = make_dead_zone_overlay(match_heat[match_heat["human"]==True])
            if dz: fig.add_trace(dz)

    human_players = match_df[match_df["human"]==True]["user_id"].unique()
    bot_players   = match_df[match_df["human"]==False]["user_id"].unique()

    if show_human_paths:
        for i, pid in enumerate(human_players):
            player_df = visible_df[visible_df["user_id"] == pid].sort_values("ts_ms")
            if player_df.empty: continue
            color = HUMAN_COLORS[i % len(HUMAN_COLORS)]
            fig.add_trace(go.Scatter(
                x=player_df["px"], y=player_df["py"],
                mode="lines+markers",
                name=f"Human {pid[:8]}",
                line=dict(color=color, width=1.5),
                marker=dict(size=4, color=color),
                opacity=path_opacity, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[player_df["px"].iloc[-1]], y=[player_df["py"].iloc[-1]],
                mode="markers",
                marker=dict(color=color, size=12, symbol="circle",
                            line=dict(color="white", width=2)),
                hovertext=f"Human: {pid[:12]} | {format_time(int(player_df['ts_ms'].iloc[-1]))}",
                hoverinfo="text", showlegend=False,
            ))

    if show_bot_paths:
        for bot_id in bot_players:
            bot_df = visible_df[visible_df["user_id"] == bot_id].sort_values("ts_ms")
            if bot_df.empty: continue
            fig.add_trace(go.Scatter(
                x=bot_df["px"], y=bot_df["py"],
                mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1),
                opacity=0.3, hoverinfo="skip", showlegend=False,
            ))

    marker_map = {}
    if show_kills:
        marker_map["Kill"]      = EVENT_STYLES["Kill"]
        marker_map["Killed"]    = EVENT_STYLES["Killed"]
        marker_map["BotKill"]   = EVENT_STYLES["BotKill"]
        marker_map["BotKilled"] = EVENT_STYLES["BotKilled"]
    if show_storm:
        marker_map["KilledByStorm"] = EVENT_STYLES["KilledByStorm"]
    if show_loot:
        marker_map["Loot"] = EVENT_STYLES["Loot"]

    for event_type, style in marker_map.items():
        edf = visible_df[visible_df["event"] == event_type]
        if edf.empty: continue
        hover = edf.apply(
            lambda r: f"{r['event']} | {'Human' if r['human'] else 'Bot'} | {r['user_id'][:12]} | {format_time(int(r['ts_ms']))}",
            axis=1,
        )
        fig.add_trace(go.Scatter(
            x=edf["px"], y=edf["py"],
            mode="markers", name=event_type,
            hovertext=hover, hoverinfo="text",
            marker=dict(size=style["size"], color=style["color"],
                        symbol=style["symbol"], opacity=1.0,
                        line=dict(width=1, color="white")),
        ))

    fig.update_xaxes(range=[0,1024], showgrid=False, visible=False, constrain="domain")
    fig.update_yaxes(range=[0,1024], showgrid=False, visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
        legend=dict(bgcolor="rgba(0,0,0,0.7)", font=dict(color="white", size=11),
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1, x=1.01, y=1),
        height=780,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw event data"):
        st.dataframe(
            visible_df[["user_id","human","event","x","z","ts_ms"]]
            .sort_values("ts_ms").reset_index(drop=True),
            use_container_width=True,
        )

    # ── PLAYBACK LOOP ─────────────────────────
    if st.session_state.playing:
        next_cursor = st.session_state.cursor + 3000
        if next_cursor >= match_duration:
            st.session_state.playing = False
            st.session_state.cursor  = match_duration
        else:
            st.session_state.cursor = next_cursor
        time.sleep(0.1)
        st.rerun()
