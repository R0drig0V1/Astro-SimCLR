# Entrenamiento contrastivo con imágenes de eventos astronómicos

Curso: El7006, primavera 2021


## Descripción

Actualmente los modelos supervisados basados en redes neuronales han tenido
un gran apogeo en las tareas de clasificación, detección, segmentación, entre otras
aplicaciones. A pesar de esto se requieren grandes volúmenes de datos para poder entrenar
estos modelos, lo que limita su uso práctico. En los últimos años los modelos no supervisados y
semi supervisados han presentado grandes avances alcanzando la performance de los
modelos supervisados utilizando una fracción de las etiquetas que estos utilizan.
Este proyecto busca explorar la utilización de SimCLR (Chen et al, 2020) un modelo en el
estado del arte en aprendizaje contrastivo y que estableció un gran avance en el aprendizaje no
supervisado. SimCLR es un modelo no supervisado que busca crear distintas aumentaciones
para una misma imágen y maximizar su similaridad, y a su vez maximizar la disimilaridad de
imágenes distintas. En este proyecto además se quiere estudiar su versión supervisada,
llamémoslo S-SimCLR (Khosla et al, 2020), en la que extiende el concepto de similaridad a
imágenes de la misma clase. Para evaluar qué tan bueno es este aprendizaje no supervisado
se entrena un clasificador lineal al final del embedding aprendido.
Este modelo no supervisado se busca aplicar en datos reales, imágenes de eventos
astronómicos obtenidos del proyecto Zwicky Transient Facility (ZTF). Esta investigación tiene
un impacto directo en el campo de la astroinformática y puede ser de utilidad para el broker
ALeRCE (http://alerce.science/) que busca automatizar la clasificación de eventos astronómicos
que serán obtenidos por el observatorio Vera C. Rubin construido en el Norte de Chile.

Este proyecto busca responder algunas preguntas como:

* ¿Cómo afecta el desbalance en SimCLR?
* ¿S-SimCLR entrenado de forma balanceada tiene mucho mejor perfomance que SimCLR?
* ¿Las aumentaciones utilizadas en SimCLR son igual de útiles en datasets astronómicos?


## Requerimientos

### Instalación del entorno
```
conda env create --name simclr --file env.yml
```
### Activación del entorno
```
conda activate simclr
```

### Incorporación del entorno a jupyter-notebook

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=simclr
```