from apps import create_app
# from .apps.email import Mailer

app = create_app()
# mailer = Mailer(app)if
if __name__ == '__main__':
    app.run(debug=True)




