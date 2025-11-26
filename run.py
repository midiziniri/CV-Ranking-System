from livereload import Server
from app import app  # import the Flask app from app.py

app.debug = True  # <--- this is important

server = Server(app.wsgi_app)
server.watch('templates/*.html')
server.watch('static/**/*.*')
server.watch('app.py')
server.serve(open_url_delay=True, port=5500)


