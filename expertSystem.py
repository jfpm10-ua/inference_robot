'''
 Sistema Experto para el guiado de un robot
 Esta clase contendrá el código creado por los alumnos de RyRDC para el control 
 y guiado de un robot móvil sobre un plano cartesiano pasando por un punto inicial
 y siguiendo una línea recta hasta un punto final

 Creado por: Diego Viejo
 el 26/09/2024
 Modificado por: 

'''
import math
import numpy as np
from segmento import *



class ExpertSystem:
    def __init__(self) -> None:
        self.objetivoAlcanzado = False
        self.segmentoObjetivo = None
        self.estado = 0
        self.es_Triangulo=False
        

    # función setObjetivo
    #   Especifica un segmento como objetivo para el recorrido del robot
    #   Este método NO debería ser modificado
    def setObjetivo(self, segmento):
        self.objetivoAlcanzado = False
        self.segmentoObjetivo = segmento


    # función tomarDecision. 
    #   Recibe una tupla de 3 valores con la pose del robot y un objeto
    #   de clase Segmento con la información del segmento a seguir
    #   
    #   Devuelve una tupla con la velocidad lineal y angular que se
    #   quiere dar al robot
    def tomarDecision(self, poseRobot):
        # Obtener la posición y orientación actual del robot
        robotX = poseRobot[0]
        robotY = poseRobot[1]
        robotTheta = poseRobot[2]

        # Coordenadas del segmento objetivo
        inicioX = self.segmentoObjetivo.getInicio()[0]
        inicioY = self.segmentoObjetivo.getInicio()[1]
        medio2X = self.segmentoObjetivo.getMedio()[0]
        medio2Y = self.segmentoObjetivo.getMedio()[1]
        finX = self.segmentoObjetivo.getFin()[0]
        finY = self.segmentoObjetivo.getFin()[1]

        # Si el segmento es una línea (tipo 1), calcular puntos medios
        if self.segmentoObjetivo.getType() == 1:
            medio2X = (inicioX + finX) / 2
            medio2Y = (inicioY + finY) / 2

            medio1X = (inicioX + medio2X) / 2
            medio1Y = (inicioY + medio2Y) / 2
        else:
            # Si es un triángulo (tipo 2), usar el punto medio definido
            medio2X = self.segmentoObjetivo.getMedio()[0]
            medio2Y = self.segmentoObjetivo.getMedio()[1]

        # Determinar el vector objetivo en función del estado actual
        if self.estado == 0:  # Estado: ir al inicio
            vectorObjetivo = [inicioX - robotX, inicioY - robotY]
        elif self.estado == 1:  # Estado: ir al final
            vectorObjetivo = [finX - robotX, finY - robotY]
        elif self.estado == 2:  # Estado: ir al punto medio 2
            vectorObjetivo = [medio2X - robotX, medio2Y - robotY]
        else:  # Estado: ir al punto medio 1
            vectorObjetivo = [medio1X - robotX, medio1Y - robotY]

        # Calcular distancia y ángulos del robot respecto al objetivo
        distancia_objetivo = np.linalg.norm(vectorObjetivo)
        angulo_objetivo = math.atan2(vectorObjetivo[1], vectorObjetivo[0])
        angulo_robot = math.radians(robotTheta)
        angulo_diferencia = angulo_objetivo - angulo_robot

        # Ajustar distancia mínima según el tipo de segmento y estado
        if self.segmentoObjetivo.getType() == 2 or (
            self.segmentoObjetivo.getType() == 1 and (self.estado == 0 or self.estado == 1)
        ):
            distancia_objetivo -= 1.5

        # Determinar velocidades del robot
        velocidad_lineal = min(3.0, distancia_objetivo)  # Máxima velocidad limitada a 3.0
        velocidad_angular = angulo_diferencia * 1.2      # Factor de ajuste para la rotación

        # Cambio de estado basado en la proximidad al objetivo
        if distancia_objetivo < 0.5:
            if self.segmentoObjetivo.getType() == 1:  # Para segmentos tipo línea
                if self.estado == 0:  # Llegó al inicio, ir al punto medio
                    self.estado = 3
                elif self.estado == 2:  # Llegó al medio, ir al final
                    self.estado = 1
                elif self.estado == 1:  # Llegó al final, objetivo completado
                    self.objetivoAlcanzado = True
                    self.estado = 0
                else:  # Llegó al punto medio 2, ir al punto medio 1
                    self.estado = 2
            elif self.segmentoObjetivo.getType() == 2:  # Para segmentos tipo triángulo
                if self.estado == 0:  # Llegó al inicio, ir al punto medio
                    self.estado = 2
                elif self.estado == 2:  # Llegó al medio, ir al final
                    self.estado = 1
                elif self.estado == 1:  # Llegó al final, objetivo completado
                    self.objetivoAlcanzado = True
                    self.estado = 0

        # Debugging: mostrar información del ángulo y otros datos relevantes
        print(
            f"angulo_diferencia: {angulo_diferencia}, angulo_objetivo: {angulo_objetivo}, "
            f"angulo_robot: {angulo_robot}, angulo_robot_grados: {robotTheta}"
        )

        # Retornar velocidades calculadas
        return (velocidad_lineal, velocidad_angular)





    
    # función esObjetivoAlcanzado 
    #   Devuelve True cuando el punto final del objetivo ha sido alcanzado. 
    #   Es responsabilidad de la alumna o alumno cambiar el valor de la 
    #   variable objetivoAlcanzado cuando se detecte que el robot ha llegado 
    #   a su objetivo. Esto se llevará a cabo en el método tomarDecision
    #   Este método NO debería ser modificado
    def esObjetivoAlcanzado(self):
        return self.objetivoAlcanzado
    def hayParteOptativa(self):
        return True
    
    