import reflex as rx
from .ml_logic import analyze_content
from .style import (
    glass_style,
    button_style,
    danger_style,
    PULSE_RAW_CSS,
    pulse_animation_style,
)


class State(rx.State):
    message: str = ""
    label: str = "READY"
    confidence: float = 0.0
    explanation: str = ""
    is_loading: bool = False
    result_color: str = "indigo"

    def set_message(self, text: str):
        self.message = text

    @rx.var
    def confidence_percent(self) -> int:
        return int(self.confidence * 100)

    # ⚠️ Must NOT be named "reset"
    def clear_results(self):
        self.message = ""
        self.label = "READY"
        self.confidence = 0.0
        self.explanation = ""
        self.result_color = "indigo"

    def run_analysis(self):
        if not self.message.strip():
            return rx.window_alert("Please enter a message to analyze.")

        self.is_loading = True
        yield

        result = analyze_content(self.message)

        self.label = result["label"]
        self.confidence = result["confidence"]
        self.explanation = result["explanation"]
        self.result_color = result["color"]

        self.is_loading = False


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("SPAM SHIELD AI", size="9", weight="bold"),
            rx.text(
                "INTELLIGENT MESSAGE SPAM DETECTION",
                size="1",
                color_scheme="gray",
                margin_bottom="2em",
            ),

            # Inject raw CSS for pulse animation
            rx.el.style(PULSE_RAW_CSS),

            rx.hstack(
                # LEFT: INPUT
                rx.box(
                    rx.vstack(
                        rx.text("MESSAGE INPUT", weight="bold"),
                        rx.text_area(
                            placeholder="Paste SMS / email content here…",
                            on_change=State.set_message,
                            value=State.message,
                            height="260px",
                            style=glass_style,
                        ),
                        rx.button(
                            "ANALYZE MESSAGE",
                            on_click=State.run_analysis,
                            is_loading=State.is_loading,
                            style=button_style,
                            width="100%",
                        ),
                        rx.button(
                            "CLEAR",
                            on_click=State.clear_results,
                            style=danger_style,
                            width="100%",
                        ),
                        spacing="3",
                    ),
                    style=glass_style,
                    padding="2em",
                    width="540px",
                ),

                # RIGHT: RESULT
                rx.box(
                    rx.vstack(
                        rx.text(
                            State.label,
                            size="6",
                            color=State.result_color,
                        ),
                        rx.text(
                            f"Confidence: {State.confidence_percent}%",
                            font_size="1.3em",
                            font_weight="bold",
                            color=State.result_color,
                        ),
                        rx.progress(
                            value=State.confidence_percent,
                            max=100,
                            width="260px",
                            height="18px",
                            color_scheme=State.result_color,
                            is_indeterminate=State.is_loading,
                            style=rx.cond(
                                State.label == "SPAM DETECTED",
                                pulse_animation_style,
                                {},
                            ),
                        ),
                        rx.text(
                            State.explanation,
                            size="2",
                            color_scheme="gray",
                            margin_top="1em",
                            text_align="center",
                        ),
                        align="center",
                        spacing="3",
                    ),
                    style=glass_style,
                    padding="2em",
                    width="360px",
                ),
                spacing="8",
                align="center",
            ),
        ),
        height="100vh",
        width="100%",
        background="radial-gradient(circle at center, #0f172a, #020617)",
    )


app = rx.App(theme=rx.theme(appearance="dark", accent_color="indigo"))
app.add_page(index, title="Spam Shield AI")
