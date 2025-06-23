from flask import render_template

def register_main_routes(app, services):
    """Registra rotas principais da aplicação"""
    
    @app.route('/')
    def index():
        return render_template('index.html')
