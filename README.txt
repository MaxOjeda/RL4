Tarea 4 Reinforcement Learning
Maximiliano Ojeda Aguila

Para replicar los experimentos se va a separar los comandos necesarios para cada Pregunta A, B y C:

Pregunta A):
En esta pregunta el archivo MainDiscreto ejecuta los tres algoritmos Q-Learning, Sarsa y Sarsa-lambda, mostrando el output solicitado en intervalos de 10

py MainDiscreto.py

Pregunta B):
Para esta pregunta de Actor critic vs SAC el código MainContinuo corresponde a AC, mientras que MainSAC corresponde a SAC. El output para cada uno son los largos de los episodios promedios en intervalos

py MainContinuo.py
py MainSAC.py

Pregunta C):
Para esta pregunta también hay dos archivos. BuscarParamsSAC, es el código que busca la mejor configuracion entre 10 opciones, entregando la que tiene menor avg length. Luego, Comparasion.py ejecuta 10 runs de la config original y 10 runs de la mejor encontrada:
py BuscarParamsSAC.py
py Comparacion.py



Nota:   1. En estos comando el "py" puede variar dependiendo de la versión de python o el entorno local por "python" o "python3".
        2. Tuve problemas con las versions de gym y gymnasium. Puede que cause un problema. Ejecute la preguntas de SAC en un entorno virtual aparte llamado "rl4", voy a adjuntar el entorno por si es necesario. Las preguntas A y B no tuvieron problema.