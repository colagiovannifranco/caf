# Cambios a Informe Manizales

## Todo el DOC
* Revisar siempre las escalas de los ejes de graficos multiples: es necesario que no cambian en graficos que estan adyacentes

## Introducción
1. Bullets resumiendo los hallazgos principales del análisis, a modo de Abstract. 
2. Grafico 1 y 2 (Salarios y pob) : solo compararía con ciduades de colombia. Primer análisis debería ser de población / salario / etc pero solo contra ciudades colombianas.
3. Brecha Salarial, grafico 3: Ciudad colombiana con población similar pero con salario mas alto.

## Conectividad Interna
4. Conectividad a internet: reveer calculo de promedios de colombia y latam, utilice muchas ciudades y deberia utilizar solo las de colombia y latam que poseo.

## Conectividad Externa
5. El indicador de acceso a mercados: esta ok, pero podría complementar con:
    * Tabla que Informe sobre cuales son las principales ciudades cerca y que caracteristicas tienen
    * Agregar salario de ciudades cercanas en tabla 6
    * Agregar los 3 principales ratios a las ciudades
    * eliminar masa salarial
    
## Trabajo y Conocimiento
6. Eliminar Duración de desempleo (Graficos 15-17)
7. Eliminar todos los graficos que muestran hombres vs mujeres: pongo inicialmente la comparación de tasa de actividad de Hombres vs mujeres, el resto de los indicadores de este segmento se deberían interpretar como si la distribución entre generos siguiera a esa tasa de actividad.
8. Graficos12-14 Desempleo etario: juntar las 3 barras en lugar de intentar 3 graficos diferentes
9. Extrapolar distribución del empleo: asalariados. Grafico 18
10. Mover grafico 18, junto con informalidad (grafico 6,7,8)
11. 1 grafico para secundario y superior terminado
12. Un grafico para %25-35 con superior terminado y % 18-25 estudiando superior (podria ser grafico en T)

## Conglomerados
13. Sacar grafico 28. Se transforma en un scater Salario vs HH
14. graficos 29, 30 , 31: CAMBIAR:
    * Masa salarial por sector vs empleo del sector en un grafico: una barra apilada para todos salarios de cada sector y otra igual para masa salarial, para cada ciudad. De esta manera, me quedan 2 barras apiladas para cada ciudad y puedo comparar la composición de empleo vs masa salarial
15. Grafico 28.B: cambia.
    * Titulo: " Salario esperado de acuerdo a composición productiva"
    * Cambiar el indicador: % por encima o por debajo del salario de referencia, no ratio (esto si estaba escrito en las anotaciones, ver informe)
    1. "No comparar salario contrafactual vs salario real"
    2. "Tema brechas entre salarios sector vs salario sector promedio nacional. y comparar esas brechas a nivel laatam/otros paises"

## Facilitación de negocios
16. Doing business (graf 32 en adelante): reveer sin son scores o dias. Poner dias, y ver si tenemos ese dato para cumplimiento de contratos.
17. Buscar base de homicidios de otros paises de latam: Mex, Bra, Arg.
18. Recalcular los promedios properati a nivel pais, haciendo promedio simple entre los valores promedio de las ciudades. Evitar calcular promedio de pais por oficina/etc ya que eso haria que bogota me suba los proemdios. Utilizar ciudades de la base de Merge que pasan el filtro. Intentar comparar con valores de Argentina y Ecuador.
19. Consultar quique por observaciones no ubicadas en ciudades en base properati. Puede que posea las geometrias de las ciudades para hacer joins y revisar si se encuentran dentro de las ciudades.
    
