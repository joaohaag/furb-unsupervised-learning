Reconhecimento Facial - utilizando OpenCV

O Projeto foi desenvolvido utilizando a ferramenta do google colab.

Para desenvolver os códigos foram utilizadas as seguintes bibliotecas:

cv2 - Biblioteca do OpenCV
random - Para embaralhar os itens da lista de imagens
os - Biblioteca utilizada para carregar as imagens para o colab
Image - Utilizada para manipulação das imagens no python
numpy

O projeto trata-se de reconhecimento facial através de um ensaio de imagens onde 70% das imagens de uma pessoa foram utilizadas para realizar o treinamento do classificador e os outros 30% são utilizados juntamente com o classificador treinado para realizar o reconhecimento.
O código realizar o treinamento e o teste utilizando de 10 a 20 componentes principais e registra a acurácia de cada execução (percetual de acerto das imagens).

Como Executar

Para executar, basta copiar o código presente no arquivo XXX e criar um novo projeto no google colab. As imagens que seão utilizadas devem ser upadas para a pasta ORL que deve ser criada no colab na área de arquivos. Após as imagens carregadas basta executar o código que o resultado da acurácia será exibido.
