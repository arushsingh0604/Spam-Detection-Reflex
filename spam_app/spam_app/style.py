# Design tokens
indigo_primary = "#6366f1"
glass_bg = "rgba(15, 23, 42, 0.6)"


glass_style = {
    "background": glass_bg,
    "backdrop_filter": "blur(20px)",
    "border": "1px solid rgba(255,255,255,0.1)",
    "border_radius": "20px",
    "box_shadow": "0 10px 40px rgba(0,0,0,0.4)",
}


button_style = {
    "background": f"linear-gradient(90deg, #4f46e5, {indigo_primary})",
    "border_radius": "12px",
    "_hover": {"transform": "scale(1.03)"},
}


danger_style = {
    "background": "linear-gradient(90deg, #dc2626, #991b1b)",
    "border_radius": "12px",
    "_hover": {"transform": "scale(1.03)"},
}


# Raw CSS for pulse animation
PULSE_RAW_CSS = """
@keyframes pulse_red {
  0% { box-shadow: 0 0 0 0 rgba(239,68,68,.6); }
  70% { box-shadow: 0 0 0 25px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
"""


pulse_animation_style = {
    "animation": "pulse_red 1.5s infinite",
}
