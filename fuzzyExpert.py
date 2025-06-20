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
        self.estado = 0  # 0: ir al inicio, 1: ir al final, 2: ir al punto medio
        
        # Variables difusas de entrada
        self.distancia = FuzzyVariable(
            universe_range=(0, 100),
            terms={
                'muy_cerca': [(0, 1), (0.5, 1), (2, 0)],
                'cerca': [(0.5, 0), (2, 1), (5, 1), (10, 0)],
                'medio': [(5, 0), (10, 1), (20, 1), (30, 0)],
                'lejos': [(20, 0), (30, 1), (50, 1), (70, 0)],
                'muy_lejos': [(50, 0), (70, 1), (100, 1)]
            }
        )
        
        self.angulo_error = FuzzyVariable(
            universe_range=(-3.14, 3.14),
            terms={
                'muy_izquierda': [(-3.14, 1), (-1.5, 1), (-0.8, 0)],
                'izquierda': [(-1.5, 0), (-0.8, 1), (-0.3, 1), (-0.1, 0)],
                'centro': [(-0.3, 0), (-0.1, 1), (0.1, 1), (0.3, 0)],
                'derecha': [(0.1, 0), (0.3, 1), (0.8, 1), (1.5, 0)],
                'muy_derecha': [(0.8, 0), (1.5, 1), (3.14, 1)]
            }
        )
        
        # Variables difusas de salida
        self.velocidad_lineal = FuzzyVariable(
            universe_range=(0, 3),
            terms={
                'parada': [(0, 1), (0.2, 1), (0.5, 0)],
                'lenta': [(0.2, 0), (0.5, 1), (1, 1), (1.5, 0)],
                'media': [(1, 0), (1.5, 1), (2, 1), (2.5, 0)],
                'rapida': [(2, 0), (2.5, 1), (3, 1)]
            }
        )
        
        self.velocidad_angular = FuzzyVariable(
            universe_range=(-1, 1),
            terms={
                'giro_izq_fuerte': [(-1, 1), (-0.6, 1), (-0.3, 0)],
                'giro_izq_suave': [(-0.6, 0), (-0.3, 1), (-0.1, 1), (0, 0)],
                'recto': [(-0.1, 0), (0, 1), (0.1, 0)],
                'giro_der_suave': [(0, 0), (0.1, 1), (0.3, 1), (0.6, 0)],
                'giro_der_fuerte': [(0.3, 0), (0.6, 1), (1, 1)]
            }
        )
        
        # Reglas difusas
        self.reglas = [
            # Reglas para control de velocidad lineal
            FuzzyRule(
                premise=[
                    ('distancia', 'muy_cerca'),
                ],
                consequence=[
                    ('velocidad_lineal', 'parada'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'cerca'),
                ],
                consequence=[
                    ('velocidad_lineal', 'lenta'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'medio'),
                ],
                consequence=[
                    ('velocidad_lineal', 'media'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'lejos'),
                ],
                consequence=[
                    ('velocidad_lineal', 'rapida'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'muy_lejos'),
                ],
                consequence=[
                    ('velocidad_lineal', 'rapida'),
                ]
            ),
            
            # Reglas para control de velocidad angular
            FuzzyRule(
                premise=[
                    ('angulo_error', 'muy_izquierda'),
                ],
                consequence=[
                    ('velocidad_angular', 'giro_izq_fuerte'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('angulo_error', 'izquierda'),
                ],
                consequence=[
                    ('velocidad_angular', 'giro_izq_suave'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('angulo_error', 'centro'),
                ],
                consequence=[
                    ('velocidad_angular', 'recto'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('angulo_error', 'derecha'),
                ],
                consequence=[
                    ('velocidad_angular', 'giro_der_suave'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('angulo_error', 'muy_derecha'),
                ],
                consequence=[
                    ('velocidad_angular', 'giro_der_fuerte'),
                ]
            ),
            
            # Reglas combinadas para mejor control
            FuzzyRule(
                premise=[
                    ('distancia', 'muy_cerca'),
                    ('angulo_error', 'centro'),
                ],
                consequence=[
                    ('velocidad_lineal', 'lenta'),
                    ('velocidad_angular', 'recto'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'cerca'),
                    ('angulo_error', 'izquierda'),
                ],
                consequence=[
                    ('velocidad_lineal', 'lenta'),
                    ('velocidad_angular', 'giro_izq_suave'),
                ]
            ),
            FuzzyRule(
                premise=[
                    ('distancia', 'cerca'),
                    ('angulo_error', 'derecha'),
                ],
                consequence=[
                    ('velocidad_lineal', 'lenta'),
                    ('velocidad_angular', 'giro_der_suave'),
                ]
            ),
        ]
        
        # Modelo de inferencia
        self.modelo = DecompositionalInference(
            and_operator='min',
            or_operator='max',
            implication_operator='Rc',
            composition_operator='max-min',
            production_link='max',
            defuzzification_operator='CoG'
        )

    def setObjetivo(self, obj):
        self.objetivoAlcanzado = False
        self.segmentoObjetivo = obj
        self.estado = 0

    def _normalizar_angulo(self, angulo):
        """Normaliza un ángulo al rango [-pi, pi]"""
        while angulo > math.pi:
            angulo -= 2 * math.pi
        while angulo < -math.pi:
            angulo += 2 * math.pi
        return angulo

    def _punto_en_triangulo(self, punto, triangulo):
        """Verifica si un punto está dentro de un triángulo"""
        def signo(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        inicio = triangulo.getInicio()
        medio = triangulo.getMedio()
        fin = triangulo.getFin()
        
        d1 = signo(punto, inicio, medio)
        d2 = signo(punto, medio, fin)
        d3 = signo(punto, fin, inicio)
        
        tiene_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        tiene_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (tiene_neg and tiene_pos)

    def tomarDecision(self, poseRobot):
        if self.segmentoObjetivo is None:
            return (0, 0)
            
        robotX, robotY, robotTheta = poseRobot[0], poseRobot[1], poseRobot[2]
        
        # Obtener coordenadas del objetivo
        inicioX, inicioY = self.segmentoObjetivo.getInicio()
        finX, finY = self.segmentoObjetivo.getFin()
        
        # Determinar punto objetivo según el estado y tipo de segmento
        if self.segmentoObjetivo.getType() == 1:  # Segmento
            if self.estado == 0:  # Ir al inicio
                objX, objY = inicioX, inicioY
            else:  # Ir al final
                objX, objY = finX, finY
        else:  # Triángulo
            if self.estado == 0:  # Ir al inicio
                objX, objY = inicioX, inicioY
            elif self.estado == 1:  # Ir al final
                objX, objY = finX, finY
            else:  # Evitar triángulo - ir al punto medio
                medioX, medioY = self.segmentoObjetivo.getMedio()
                objX, objY = medioX, medioY
        
        # Calcular distancia y ángulo al objetivo
        dx = objX - robotX
        dy = objY - robotY
        distancia_obj = math.sqrt(dx*dx + dy*dy)
        angulo_objetivo = math.atan2(dy, dx)
        angulo_robot = math.radians(robotTheta)
        error_angular = self._normalizar_angulo(angulo_objetivo - angulo_robot)
        
        # Para triángulos, verificar si necesitamos evitar el área
        if self.segmentoObjetivo.getType() == 2 and self.estado == 0:
            # Si estamos muy cerca del triángulo, cambiar a modo evasión
            if self._punto_en_triangulo((robotX, robotY), self.segmentoObjetivo):
                self.estado = 2  # Ir al punto medio para evitar
                medioX, medioY = self.segmentoObjetivo.getMedio()
                dx = medioX - robotX
                dy = medioY - robotY
                distancia_obj = math.sqrt(dx*dx + dy*dy)
                angulo_objetivo = math.atan2(dy, dx)
                error_angular = self._normalizar_angulo(angulo_objetivo - angulo_robot)
        
        # Aplicar inferencia difusa
        try:
            resultado = self.modelo.compute(
                self.reglas,
                distancia=distancia_obj,
                angulo_error=error_angular
            )
            
            V = resultado.get('velocidad_lineal', 0)
            W = resultado.get('velocidad_angular', 0)
        except:
            # Fallback en caso de error en inferencia difusa
            V = min(3.0, distancia_obj * 0.5)
            W = error_angular * 0.8
        
        # Limitar velocidades
        V = max(0, min(3, V))
        W = max(-1, min(1, W))
        
        # Gestión de estados
        if distancia_obj < 0.5:
            if self.segmentoObjetivo.getType() == 1:  # Segmento
                if self.estado == 0:
                    self.estado = 1  # Del inicio al final
                else:
                    self.objetivoAlcanzado = True
                    self.estado = 0
            else:  # Triángulo
                if self.estado == 0:
                    self.estado = 2  # Del inicio al punto medio
                elif self.estado == 2:
                    self.estado = 1  # Del punto medio al final
                else:
                    self.objetivoAlcanzado = True
                    self.estado = 0
        
        return (V, W)
    
    def esObjetivoAlcanzado(self):
        return self.objetivoAlcanzado
    
    def hayParteOptativa(self):
        return True