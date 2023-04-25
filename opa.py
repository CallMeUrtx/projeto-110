import cv2
import tensorflow as tf

# Carrega o modelo gerado pelo Teachable Machine
model_path = 'caminho/para/o/modelo.pb'
model = tf.keras.models.load_model(model_path)

# Define as classes do jogo
classes = ['pedra', 'papel', 'tesoura']

# Captura imagens da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Redimensiona a imagem para o tamanho usado no treinamento do modelo
    img = cv2.resize(frame, (224, 224))
    
    # Normaliza os valores dos pixels da imagem
    img = img / 255.0
    
    # Faz a previsão da classe da imagem capturada
    pred = model.predict(img.reshape(1, 224, 224, 3))[0]
    index = tf.argmax(pred).numpy()
    jogada_computador = classes[index]
    
    # Define a jogada do usuário
    jogada_usuario = input("Digite sua jogada (pedra, papel ou tesoura): ")
    
    # Define quem venceu o jogo
    if jogada_usuario == jogada_computador:
        resultado = "Empate!"
    elif jogada_usuario == "pedra" and jogada_computador == "tesoura":
        resultado = "Você ganhou!"
    elif jogada_usuario == "tesoura" and jogada_computador == "papel":
        resultado = "Você ganhou!"
    elif jogada_usuario == "papel" and jogada_computador == "pedra":
        resultado = "Você ganhou!"
    else:
        resultado = "Você perdeu!"
        
    # Exibe o resultado
    print(f"Você jogou {jogada_usuario} e o computador jogou {jogada_computador}. {resultado
