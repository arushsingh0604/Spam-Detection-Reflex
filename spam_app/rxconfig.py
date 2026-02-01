<<<<<<< HEAD
import reflex as rx

config = rx.Config(
    app_name="spam_app",
    api_url="http://15.207.87.135:8000",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
=======
import reflex as rx

config = rx.Config(
    app_name="spam_app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
>>>>>>> ae3abbbc2e41e8ddf09d614a1558e0a879b3aacb
)