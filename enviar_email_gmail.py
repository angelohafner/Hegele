import smtplib
import email.message
import os

def enviar_email():
    corpo_email = """
    <p>Parágrafo1</p>
    <p>Parágrafo2</p>
    """

    msg = email.message.Message()
    msg['Subject'] = "Assunto"
    msg['From'] = 'angelo.hafner@gmail.com'
    msg['To'] = 'angelo@hafner-inc.com'
    password = os.environ.get('EMAIL_PASSWORD')  # Obtenha a senha a partir de uma variável de ambiente
    msg.add_header('Content-Type', 'text/html')
    msg.set_payload(corpo_email)

    s = smtplib.SMTP('smtp.gmail.com: 587')
    s.starttls()
    # Login Credentials for sending the mail
    s.login(msg['From'], password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string().encode('utf-8'))
    print('Email enviado')

if __name__ == '__main__':
    enviar_email()
