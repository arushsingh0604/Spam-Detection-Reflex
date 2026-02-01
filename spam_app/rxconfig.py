import reflex as rx

config = rx.Config(
    app_name="spam_app",
    api_url="http://15.207.87.135:8000",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)