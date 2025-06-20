'''
 Sistema Experto Difuso para el guiado de un robot
 Esta clase contendrá el código creado por los alumnos de RyRDC para el control 
 y guiado de un robot móvil sobre un plano cartesiano para recorrer diferentes 
 objetivos utilizando un esquema de sistema experto difuso.
 
 Para implementar el sistema experto difuso hay que instalar la librería
 https://jdvelasq.github.io/fuzzy-expert/

 Creado por: Diego Viejo
 el 24/10/2024
 Modificado por: Diego Viejo. 

'''

import numpy as np
import math

from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.inference import DecompositionalInference

from segmento import *

class FuzzySystem:
    def __init__(self) -> None:
        self.objetivoAlcanzado = False
        self.segmentoObjetivo = None
        #Completar definiendo las variables difusas y las reglas difusas del sistema experto.
        #Construcción del modelo difuso

    # función setObjetivo
    #   Especifica un objetivo que debe ser recorrido por el robot
    def setObjetivo(self, obj):
        self.objetivoAlcanzado = False
        self.segmentoObjetivo = obj

    def tomarDecision(self, poseRobot):
        V = W = 0
        #uso del modelo difuso para objeter V y W
        return (V, W)
    
    # función esObjetivoAlcanzado 
    #   Devuelve True cuando el punto final del objetivo ha sido alcanzado. 
    #   Es responsabilidad de la alumna o alumno cambiar el valor de la 
    #   variable objetivoAlcanzado cuando se detecte que el robot ha llegado 
    #   a su objetivo. Esto se llevará a cabo en el método tomarDecision
    #   Este método NO debería ser modificado
    def esObjetivoAlcanzado(self):
        return self.objetivoAlcanzado
    
    # función hayParteOptativa.
    #   Deberá devolver True si la parte optativa ha sido implementada, es decir, si se consideran objetivos de tipo triángulo
    def hayParteOptativa(self):
        return False
