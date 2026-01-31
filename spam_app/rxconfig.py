import reflex as rx

config = rx.Config(
    app_name="spam_app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)