import numpy as np
import pandas as pd
import streamlit as st

from modules.scenarios import SCENARIOS
from modules.solve_2x2 import graphical_data
from modules.solver import solve_game
from utils.data_validator import validate_matrix

# ---------------------------------------------------------
# 1. Page configuration & styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AoMiGT | Matrix Game Solver",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, SLATE, TEAL, LINE, INK = "#1F3A5F", "#5B6675", "#2A7F8E", "#D9DEE6", "#1B2430"

st.markdown(f"""
<style>
    html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: {INK}; }}
    #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
    .block-container {{ padding-top: 2.4rem; max-width: 1120px; }}
    .app-header {{ border-bottom: 1px solid {NAVY}; padding-bottom: 14px; margin-bottom: 26px; }}
    .app-kicker {{ font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; color: {SLATE}; margin-bottom: 6px; }}
    .app-title {{ font-family: Georgia, 'Times New Roman', serif; font-size: 2.05rem; font-weight: 600; color: {NAVY}; margin: 0; line-height: 1.2; }}
    .app-sub {{ font-size: .98rem; color: {SLATE}; margin-top: 6px; }}
    h2, h3 {{ font-family: Georgia, 'Times New Roman', serif; color: {NAVY}; font-weight: 600; letter-spacing: 0; }}
    .sec {{ font-family: Georgia, 'Times New Roman', serif; font-size: 1.28rem; font-weight: 600; color: {NAVY};
            border-bottom: 1px solid {LINE}; padding-bottom: 6px; margin: 2.1rem 0 .9rem; }}
    .sec .no {{ color: {TEAL}; margin-right: .55rem; font-weight: 600; }}
    .tag {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {SLATE}; margin-bottom: 4px; }}
    [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-top: 3px solid {NAVY};
                               border-radius: 2px; padding: 14px 18px; }}
    [data-testid="stMetricLabel"] p {{ color: {SLATE}; font-size: .8rem; letter-spacing: .05em; text-transform: uppercase; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-family: Georgia, 'Times New Roman', serif; font-variant-numeric: tabular-nums; }}
    [data-testid="stSidebar"] {{ background: #F6F8FA; border-right: 1px solid {LINE}; }}
    [data-testid="stSidebar"] h3 {{ font-size: 1.05rem; }}
    .stButton > button[kind="primary"] {{ background: {NAVY}; border: 1px solid {NAVY}; border-radius: 2px;
                                         font-weight: 500; letter-spacing: .04em; padding: .55rem 1.8rem; }}
    .stButton > button[kind="primary"]:hover {{ background: #2B4F7E; border-color: #2B4F7E; }}
    [data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 2px; }}
    .note {{ border-left: 3px solid {LINE}; background: #F8FAFC; padding: 10px 16px; margin: 10px 0; font-size: .94rem; }}
    .note.ok {{ border-left-color: #2E7D5B; }}
    .note.warn {{ border-left-color: #B7791F; }}
    .note.err {{ border-left-color: #B42318; }}
    .method {{ color: {SLATE}; font-size: .9rem; margin: 10px 0 0; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {SLATE}; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-header"><div class="app-kicker">Applications of Matrices in Game Theory</div>'
    '<div class="app-title">AoMiGT Matrix Game Solver</div>'
    '<div class="app-sub">Two-player zero-sum games: saddle points, mixed strategies and the value of the game</div></div>',
    unsafe_allow_html=True,
)


def section(no, title):
    st.markdown(f'<div class="sec">{('<span class="no">' + no + '</span>') if no else ''}{title}</div>', unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def vec(p):
    return r"\left(" + r",\ ".join(f"{float(t):.4f}" for t in p) + r"\right)"


@st.cache_data(show_spinner=False, max_entries=256)
def cached_solve(shape, flat):
    """Identical matrices are solved once and shared between all visitors."""
    return solve_game(np.array(flat, dtype=float).reshape(shape))


def short(names):
    """Short axis labels: the part before the colon (e.g. "X1: Patrol" -> "X1")."""
    return [n.split(":")[0] for n in names]


# ---------------------------------------------------------
# 2. Sidebar: choose the game
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Game setup")
    mode = st.radio("Input mode", ["Real-world scenario library", "Custom m × n matrix"])
    st.markdown("---")

    scenario = None
    if mode.startswith("Real-world"):
        titles = [s.title for s in SCENARIOS]
        chosen = st.radio("Scenario", titles)
        scenario = SCENARIOS[titles.index(chosen)]
        A0 = scenario.A.copy()
        row_names, col_names = list(scenario.row_names), list(scenario.col_names)
    else:
        m_in = st.number_input("Rows m (Player 1 strategies)", min_value=2, max_value=8, value=3, step=1)
        n_in = st.number_input("Columns n (Player 2 strategies)", min_value=2, max_value=8, value=3, step=1)
        m_in, n_in = int(m_in), int(n_in)
        A0 = np.zeros((m_in, n_in), dtype=float)
        row_names = [f"X{i + 1}" for i in range(m_in)]
        col_names = [f"Y{j + 1}" for j in range(n_in)]
        st.caption("Player 1 (rows) maximises the payoff a_ij, Player 2 (columns) minimises it.")

m_rows, n_cols = A0.shape

# ---------------------------------------------------------
# Helper plots
# ---------------------------------------------------------
def strategy_chart(probs, labels, color):
    """Light native bar chart (no heavy plotting library)."""
    df = pd.DataFrame({"Probability": np.asarray(probs, dtype=float)}, index=list(labels))
    st.bar_chart(df, color=color, height=260)


def envelope_chart(g, player, labels_other):
    """Graphical method: expected-payoff lines for each opponent pure strategy and the envelope."""
    data = {str(labels_other[k]): line for k, line in enumerate(g["lines"])}
    data["Envelope"] = g["envelope"]
    df = pd.DataFrame(data, index=np.round(np.asarray(g["t"], dtype=float), 4))
    df.index.name = "Probability of the first strategy"
    st.line_chart(df, height=300)
    st.caption(f"Optimum: t* = {g['t_opt']:.4f}, value = {g['v_opt']:.4f}")


# ---------------------------------------------------------
# 3. Matrix input and computation
# ---------------------------------------------------------
section("", "Payoff matrix")
st.latex(r"A = (a_{ij})")
if scenario is not None:
    st.caption(f"Payoff unit: {scenario.unit}. Rows = {scenario.row_player}, columns = {scenario.col_player}. "
               "You may edit the numbers.")

default_df = pd.DataFrame(A0, columns=col_names, index=row_names)
edited_df = st.data_editor(
    default_df,
    key=f"editor_{mode}_{scenario.key if scenario else 'custom'}_{m_rows}x{n_cols}",
    num_rows="fixed",
)
solve_btn = st.button("Solve game", type="primary")

ok, msg, A = validate_matrix(edited_df.to_numpy())
if not ok:
    st.error(msg)
else:
    signature = (A.shape, tuple(A.flatten().tolist()))
    if solve_btn:
        try:
            st.session_state["report"] = (signature, cached_solve(A.shape, tuple(A.flatten().tolist())))
        except Exception as exc:  # show the real reason instead of failing silently
            st.session_state.pop("report", None)
            st.error(f"The solver failed: {exc}")

    stored = st.session_state.get("report")
    if stored is None or stored[0] != signature:
        st.info("Press **Solve game** to compute the solution of the current matrix.")
    else:
        rep = stored[1]
        sp = rep.saddle
        section("", "Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Value of the game v", f"{rep.value:.4f}")
        c2.metric("Maximin (lower value)", f"{sp['maximin']:.4f}")
        c3.metric("Minimax (upper value)", f"{sp['minimax']:.4f}")
        st.markdown(f'<div class="method">Method used: <b>{rep.method}</b></div>', unsafe_allow_html=True)

        # --- Step 1: maximin / minimax
        section("Step 1", "Maximin and minimax")
        table = pd.DataFrame(A, index=row_names, columns=col_names)
        table["Row min"] = A.min(axis=1)
        table.loc["Column max"] = list(A.max(axis=0)) + [np.nan]
        st.dataframe(table)
        if sp["has_saddle_point"]:
            pts = ", ".join(f"({row_names[i].split(':')[0]}, {col_names[j].split(':')[0]})" for i, j in sp["saddle_points"])
            note(f"Maximin = Minimax = {sp['value']:.4f}. Pure saddle point(s): {pts}.", "ok")
        else:
            note(f"Maximin ({sp['maximin']:.4f}) &lt; Minimax ({sp['minimax']:.4f}): no pure saddle point, "
                 "so optimal play requires mixed strategies (Minimax Theorem).", "warn")

        # --- Step 2: dominance
        section("Step 2", "Elimination of strictly dominated strategies")
        for line in rep.dominance["steps"]:
            st.write("• " + line)
        red = rep.dominance["reduced"]
        st.write(f"Reduced matrix ({red.shape[0]} × {red.shape[1]}):")
        st.dataframe(pd.DataFrame(
            red,
            index=[row_names[i] for i in rep.dominance["row_idx"]],
            columns=[col_names[j] for j in rep.dominance["col_idx"]],
        ))

        # --- Step 3: solution
        section("Step 3", "Optimal strategies")
        if rep.two_by_two is not None:
            for line in rep.two_by_two.steps:
                st.write("• " + line)

        st.latex(r"x^{*} = " + vec(rep.x) + r",\qquad y^{*} = " + vec(rep.y) + rf",\qquad v = {rep.value:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Player 1 - optimal mixed strategy x*")
            strategy_chart(rep.x, short(row_names), NAVY)
        with col2:
            st.caption("Player 2 - optimal mixed strategy y*")
            strategy_chart(rep.y, short(col_names), TEAL)

        t1, t2 = st.columns(2)
        t1.dataframe(pd.DataFrame({"Strategy": row_names, "x*": rep.x.round(6)}))
        t2.dataframe(pd.DataFrame({"Strategy": col_names, "y*": rep.y.round(6)}))

        # --- Step 4: verification
        section("Step 4", "Verification of optimality")
        ver = rep.verification
        v1, v2 = st.columns(2)
        v1.write("Player 1's expected payoff against each pure column, (xᵀA)ⱼ — must all be ≥ v:")
        v1.dataframe(pd.DataFrame({"Column": col_names, "(xᵀA)ⱼ": ver["col_payoffs"].round(6)}))
        v2.write("Player 1's expected payoff of each pure row against y*, (Ay)ᵢ — must all be ≤ v:")
        v2.dataframe(pd.DataFrame({"Row": row_names, "(Ay)ᵢ": ver["row_payoffs"].round(6)}))
        if ver["is_optimal"]:
            note(f"Verified: min (xᵀA)ⱼ = {ver['p1_guarantee']:.4f} ≥ v = {rep.value:.4f} ≥ "
                 f"max (Ay)ᵢ = {ver['p2_guarantee']:.4f}, and xᵀAy = {ver['xAy']:.4f}.", "ok")
        else:
            note("The verification failed: (x*, y*) is not an optimal pair.", "err")

        # --- Graphical method (2 x n or m x 2)
        gdata = graphical_data(A)
        if gdata:
            with st.expander("Graphical method (2 × n or m × 2 games)"):
                if "row" in gdata:
                    envelope_chart(gdata["row"], "row", col_names)
                if "col" in gdata:
                    envelope_chart(gdata["col"], "col", row_names)

        # --- Cross-check
        section("Step 5", "Cross-check of the value of the game")
        cross = pd.DataFrame({
            "Method": ["Main solution", "LP (SciPy HiGHS)", "Own simplex implementation"],
            "Value v": [rep.value, rep.lp.value, rep.simplex.value],
            "Optimal pair verified": [ver["is_optimal"], rep.lp_verification["is_optimal"],
                                      rep.simplex_verification["is_optimal"]],
        })
        st.dataframe(cross)
        if rep.values_agree:
            note("All methods give the same value of the game (Minimax Theorem).", "ok")
        else:
            note("The methods disagree on the value; check the input matrix.", "err")
