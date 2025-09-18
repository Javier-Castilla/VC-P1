# Práctica 1 - Visión por Computador (2025 - 2026)
# Autores
- Asmae Ez Zaim Driouch
- Javier Castilla Moreno
# Bibliotecas utilizadas
[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%43FF6400?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
# Cómo usar
## Primer paso: clonar este repositorio
```bash
git clone "https://github.com/Javier-Castilla/VC-P1"
```
## Segundo paso: Activar tu envinroment e instalar dependencias
> [!NOTE]
> Todas las dependencias pueden verse en [este archivo](envinronment.yml). Si se desea, puede crearse un entorno de Conda con dicho archivo.

Si se opta por crear un nuevo `Conda envinronment` a partir del archivo expuesto, es necesario abrir el `Anaconda Prompt` y ejecutar lo siguiente:

```bash
conda env create -f environment.yml
```

Posteriormente, se activa el entorno:

```bash
conda activate VC_P1
```

## Tercer paso: ejecutar el cuaderno
Finalmente, abriendo nuestro IDE favorito y teniendo instalado todo lo necesario para poder ejecutar notebooks, se puede ejecutar el cuaderno de la práctica [Práctica1.ipynb](Práctica1.ipynb) seleccionando el envinronment anteriormente creado.

> [!IMPORTANT]
> Todos los bloques de código deben ejecutarse en órden, de lo contrario, podría ocasionar problemas durante la ejecución del cuaderno.

# Tarea 2: Imagen al estilo Mondrian
Se ha generado una imagen al estilo `Mondrian` haciendo uso de las utilidades que presenta la biblioteca `OpenCV`.

Concretamente, se ha tomado como referencia la siguiente imagen:

<img src="https://www.descubrirelarte.es/wp-content/uploads/2020/11/Composicion-con-amarillo-rojo-negro-azul-y-gris-por-Piet-Mondrian-1920-oleo-sobre-lienzo-595-x-595-cm-La-Haya-Gemeentemuseum..jpg">

La manera de proceder ha sido sencilla, se han guardado en una lista de python las distintas coordenadas necesarias para dibujar los distintos rectángulos que conforman la imagen. Junto a las coordenadas, se establece el color que tendrá cada rectángulo.

Una vez guardadas las diferentes coordenadas y colores, se recorre la lista y se dibuja sobre una imagen generada inicialmente en blanco cada rectángulo de la siguiente manera:

```python
for i, rectangle in enumerate(rectangles):
    cv2.rectangle(img, rectangle[0], rectangle[1], rectangle[2], -1)
    cv2.rectangle(img, rectangle[0], rectangle[1], (0, 0, 0), 5)
```

> [!NOTE]
> La función `cv2.rectangle` es la encargada de dicujar cada rectángulo pasándole la imagen, las coordenadas de las esquinas superior izquierda e inferior derecha, su color y su ancho, siendo `-1` el valor para rellenar el rectángulo

Posteriormente, se muestra la imagen y se guarda en disco de la siguiente manera:

```python
plt.imshow(img)
plt.show()
cv2.imwrite('imgs/mondrian.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

> [!NOTE]
> Destacar que es necesario pasar la imagen a BGR antes de guardarla en disco haciendo uso de OpenCV para una correcta visualización posterior. Esto seguirá presente a lo largo de la práctica.

La imagen resultante es la siguiente:

<img src=imgs/mondrian.jpg>

# Tarea 3: editar los diferentes planos de una imagen
Para esta tarea, se han editado los diferentes planos tanto de una imagen guardada en disco como los fotogramas de un vídeo en vivo tomados desde la webcam del ordenador.
Con el fin de que dichas modificaciones sean reutilizables y aplicables a diferentes imágenes o fotogramas, se ha realizado una clase para esta tarea, permitiendo aplicar filtros y máscaras de una manera sencilla y cómoda (`Tarea3`).

En dicha existen métodos estáticos que permiten invertir los diferentes colores de una imagen (R, G, B) así como el negativo de la misma o en su defecto, modificar individualmente y de manera personalizada los diferentes canales. En resumen, permite las siguientes operaciones:
- Imagen en negativo
- Invertir un canal específico
- Modificar canales de manera personalizada

## Modificación de los diferentes canales de una imagen leída de disco
Para lograr esto, se ha seleccionado una imagen y se ha cargado en memoria desde el disco con el siguiente código:

```python
image = cv2.imread('imgs/happy_hamster.jpg', cv2.IMREAD_COLOR_RGB)
plt.imshow(image)
plt.show()
```

Mostrándose la siguiente imagen:

<img src="imgs/happy_hamster.jpg" width=800 height=800>

> [!NOTE]
> La función `cv2.imread` es la encargada de leer la imagen de disco pasadas una ruta y la forma de lectura, que en este caso ha sido lectura en `RGB`, pues originalmente `OpenCV` lee las imágenes en `BGR`.

A continuación, se han invertido los colores de la imagen haciendo uso de la clase `Tarea3` nombrada anteriormente.

Se presenta el código de dicho método:

```python
@staticmethod
def negative_image(image):
    return 255 - image
```

Simplemente, se resta al `valor máximo (255)` el valor de cada píxel de la imagen, obteniendo así su complementario. El resultado se muestra con el siguiente código:

```python
plt.imshow(img := Tarea3.apply(image, Tarea3.NEGATIVE))
plt.show()
cv2.imwrite("imgs/happy_hamster_negative.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

Y ésta es la imagen resultante:

<img src="imgs/happy_hamster_negative.jpg">

Aplicando ésta misma estrategia, se ha invertido el canal `verde (G)` de la imagen original, resultando en el siguiente código:

```python
plt.imshow(img := Tarea3.apply(image, Tarea3.INVERT_GREEN))
plt.show()
cv2.imwrite("imgs/happy_hamster_green_negative.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

Resultando en la siguiente imagen:

<img src="imgs/happy_hamster_green_negative.jpg">

El método utilizado para ello ha sido el siguiente:

```python
@staticmethod
def invert_green(image):
    image = image.copy()
    image[:,:,1] = 255 - image[:,:,1]
    return image
```

En él, se observa como se aplica la misma estrategia inicial, pues se le resta al `mñaximo valor (255)` el valor de cada píxel en el `canal verde (G)`.

> [!NOTE]
> Para invertir el resto de colores se aplica la misma estrategia en el `canal deseado`.

Finalmente, se ha editado de manera aleatoria la imagen original, `invirtiendo el canal rojo (R)`, estableciendo al `12% de su valor original el canal verde (G)` y `omitiendo completamente el canal azul (B)`. Para ello se ha hecho uso del siguiente código:

```python
plt.imshow(img := Tarea3.change_color_percentage(image, -1, 0.12, 0))
plt.show()
cv2.imwrite("imgs/happy_hamster_random.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

Resultando en la siguiente imagen:

<img src="imgs/happy_hamster_random.jpg">

Para lograr esto, se ha hecho uso del siguiente método:

```python
@staticmethod
def change_color_percentage(image, r=1, g=1, b=1):
    image = image.copy()
    image[:,:,0] = (image[:,:,0] * r) if r != -1 else 255 - image[:,:,0]
    image[:,:,1] = (image[:,:,1] * g) if g != -1 else 255 - image[:,:,1]
    image[:,:,2] = (image[:,:,2] * b)if b != -1 else 255 - image[:,:,2]
    return image
```

Como se puede observar, se pasan por parámetros tanto la `imagen` como el `porcentaje del valor` que tendrá cada canal en la nueva imagen, `siendo -1 la inversión de dicho canal`.

## Modificación de los diferentes canales de los fotogramas de un vídeo en vivo tomado desde la webcam
Para lograr obtener un vídeo en vivo a través de la webcam, al igual que para leer una imagen de disco, se ha usado la biblioteca `OpenCV`, concretamente el siguiente método:

```python
video = cv2.VideoCapture(0)
```

Ésto, con la ayuda de un bucle infinito, permite leer los diferentes fotogramas del vídeo en tiempo real tomado por la webcam de la siguiente manera:

```python
while True:
    ret, frame = video.read()
```

> [!NOTE]
> En la variable `ret`se indica si se ha leído un fotograma, mientras que en `frame` se guarda el fotograma en caso de ser leído correctamente.

En este caso, se han añadido controles para cambiar entre la inversión total de canales de los fotogramas o en consecuencia, la inversión de un canal concreto. Los controles son los siguientes:
- `1` -> Negativo
- `2` -> Verde invertido
- `3` -> Azul invertido
- `4` -> Rojo invertido

Al elegir la máscara que queramos aplicar a la imagen, se mostrará en tiempo real, permitiéndo ver el vídeo tomado por la webcam de maneras singulares.

A continuación, se muestra una tabla con las cuatro modificaciones posibles:

<table align"center>
    <td width="25%">
        <img src="imgs/video_negative.jpg">
    </td>
    <td width="25%">
        <img src="imgs/video_red_negative.jpg">
    </td>
    <td width="25%">
        <img src="imgs/video_green_negative.jpg">
    </td>
    <td width="25%">
        <img src="imgs/video_blue_negative.jpg">
    </td>
</table>

Para la definición de los controles, se ha usado el siguiente fragmento de código:

```python
pressed_key = cv2.waitKey(20) & 0xFF

if pressed_key == 27:
    break
elif pressed_key in methods_map:
    current_method = methods_map[pressed_key]
```

En el código anterior, 27 es el valor `ASCII` asignado a la tecla `ESC`. Si se ha detectado esta tecla, se termina la ejecución. Encambio, si se ha detectado cualquier tecla del `1 al 4`, se aplica al fotograma la modificación de canales correspondiente.

> [!NOTE]
> Para la modificación de los diferentes canales de cada fotograma, se ha seguido el mismo procedimiento que con la imagen leída de disco, es decir, se ha hecho uso de las utilidades desarrolladas en la clase `Tarea3`.
# Bibliografía
- [Repositorio base y enunciado de ésta práctica](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P1)
- [cv2.minMaxLoc() - OpenCV Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga7622c466c628a75d9ed008b42250a73f)
- [Resizing and Rescaling Images with OpenCV](https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/)
- [Numpy sorting and searching Documentation](https://numpy.org/doc/2.1/reference/routines.sort.html)
- [Numpy indexes](https://numpy.org/doc/2.1/reference/routines.indexing.html)