import cv2
import random
import os
import numpy as np
#Usado para carregamento de imagem no python
from PIL import Image

def getImagemComId():
  #Carrega Imagens
  caminhos = [os.path.join('ORL', f) for f in os.listdir('ORL')] # Deve-se criar uma pasta no colab com o nome ORL e colocar as imagens na pasta.

  #Embaralha itens da lista
  random.shuffle(caminhos)

  #Declara Listas
  idsTreino = []
  idsTeste = []
  facesTreino = []
  facesTeste = []

  #Percorre as imagens
  for caminhoImagem in caminhos:
    #atribui a imagem convertendo para níveis de cinza
    imagemFace = Image.open(caminhoImagem).convert('L') #Carrega as imagens e convert para tons de cinza
    imegem_resize = imagemFace.resize((80,80)) #redimenciona a imgem devido
    imagemNP = np.array(imegem_resize, 'uint8')  #inteiro sem sinal
    #atribui o id com base na label da pessoa
    id = int(os.path.split(os.path.split(caminhoImagem)[-1].split('_')[1])[-1].split('.')[0])
    #Conta quantas vezes a label/id está presente na lista. Se já tiver 7 coloca na lista de Teste, caso contrário treino.
    if idsTreino.count(id) < 7:
      idsTreino.append(id)
      facesTreino.append(imagemNP)    
    else:
      idsTeste.append(id)
      facesTeste.append(imagemNP)
  return np.array(idsTreino), np.array(idsTeste), facesTreino, facesTeste

idsTreino, idsTeste, facesTreino, facesTeste = getImagemComId()

for k in range(10, 21, 1):
  eigenface = cv2.face.EigenFaceRecognizer_create(k)

  eigenface.train(facesTreino, idsTreino)
  eigenface.write('classificadorEigen_'+str(k)+'.yml')

  detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  reconhecedor = cv2.face.EigenFaceRecognizer_create()
  reconhecedor.read("classificadorEigen_"+str(k)+".yml")

  totalAcertos = 0
  percentualAcerto = 0.0
  i = 0

  for imagemTeste in facesTeste:
    facesDetectadas = detectorFace.detectMultiScale(imagemTeste)
    for (x, y, l, a) in facesDetectadas:
      idprevisto, confianca = reconhecedor.predict(imagemTeste)
      idatual = idsTeste[i]
      if idprevisto == idatual:
        totalAcertos += 1
    i += 1
  percentualAcerto = (totalAcertos / len(idsTeste)) * 100
  print(str(k) + " componentes principais, acurácia: " + str(percentualAcerto) +"%")
