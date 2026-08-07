# src/utils/notifier.py
def send_fraud_alert_email(
    merchant_email: str, transaction_id: str, fraud_score: float
):
    # Simulation d'envoi d'e-mail avec SMTP ou API transactionnelle (SendGrid/Mailgun)
    email_body = f"""
    Sujet : [Alerte Fraude] Transaction Suspecte Bloquée
    
    Bonjour,
    
    Une transaction suspecte a été détectée et bloquée sur votre boutique :
    - ID Transaction : {transaction_id}
    - Score de suspicion : {round(fraud_score * 100, 2)}%
    
    Aucun débit n'a été effectué sur le compte de votre client.
    
    L'équipe Sécurité MLOps.
    """

    print(f"E-mail envoyé avec succès à {merchant_email} :")
    print(email_body)
    return True
