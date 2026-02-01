import reflex as rx

# --- Glassmorphism & Effects ---
glass_style = {
    "background": "rgba(15, 23, 42, 0.6)",
    "backdrop_filter": "blur(20px) saturate(180%)",
    "border": "1px solid rgba(255, 255, 255, 0.1)",
    "border_radius": "24px",
    "box_shadow": "0 8px 32px 0 rgba(0, 0, 0, 0.4)",
}

button_style = {
    "background": "linear-gradient(90deg, #4f46e5, #6366f1)",
    "border_radius": "12px",
    "transition": "all 0.3s ease",
    "_hover": {
        "transform": "translateY(-2px)",
        "box_shadow": "0 0 20px rgba(99, 102, 241, 0.4)",
    },
}

# --- Expert-Level "Threat" Pulse Animation ---
# This creates a radiating red ring when a threat is detected
pulse_animation = rx.keyframes({
    "0%": {"box_shadow": "0 0 0 0px rgba(239, 68, 68, 0.7)"},
    "70%": {"box_shadow": "0 0 0 25px rgba(239, 68, 68, 0)"},
    "100%": {"box_shadow": "0 0 0 0px rgba(239, 68, 68, 0)"},
})