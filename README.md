# azure-ml-projects

> Este repositorio contiene proyectos desarrollados por estudiantes del Bootcamp Azure Data Science.

### Lineamientos del Proyecto Final

#### Descripción del Proyecto:
El proyecto final consistirá en el diseño, desarrollo e implementación de una solución de aprendizaje automático utilizando Azure Machine Learning. Los estudiantes deberán aplicar todos los conocimientos adquiridos durante el bootcamp para crear un modelo que resuelva un problema real de ciencia de datos, desde la exploración de datos hasta la implementación y supervisión del modelo en un entorno de producción (este último es opcional).

#### Objetivos:
- Demostrar la capacidad de diseñar y preparar una solución completa de aprendizaje automático.
- Aplicar técnicas de exploración de datos y entrenamiento de modelos.
- Implementar canalizaciones y ejecutar trabajos en Azure Machine Learning.
- Administrar y supervisar el rendimiento del modelo en producción.

#### Requisitos:
- Utilización de Azure Machine Learning para la creación y gestión del proyecto.
- Implementación de al menos una canalización completa que incluya la preparación de datos, entrenamiento del modelo y evaluación del modelo.
- Uso de MLflow para el seguimiento y gestión de experimentos.
- Presentación de un informe detallado que documente el proceso seguido, los resultados obtenidos y las decisiones tomadas.

#### Recomendaciones:
- Elegir un problema de ciencia de datos que sea relevante y tenga datos accesibles para el análisis.
- Dividir el trabajo en etapas manejables: definición del problema, recopilación y limpieza de datos, entrenamiento del modelo, evaluación, implementación y monitoreo (opcional).
- Documentar todas las decisiones y procesos de manera clara y concisa.
- Realizar pruebas exhaustivas para validar el modelo antes de su implementación.

### Instrucciones para los Participantes

1. **Fork del Repositorio**:
   - Cada equipo deberá hacer un fork del repositorio principal `azure-ml-projects` en GitHub.

2. **Crear Carpeta del Equipo**:
   - Dentro del directorio `projects/`, crear una carpeta con el nombre de su equipo (por ejemplo, `team1_project`).

3. **Organizar Archivos**:
   - Seguir la estructura proporcionada para organizar su trabajo dentro de la carpeta del equipo. La estructura recomendada es:
     ```
     teamX_project/
     |── README.md
     |── .gitignore
     ├── data/
     │   └── raw_data.csv # opcional
     ├── notebooks/
     │   ├── data_exploration.ipynb
     │   ├── model_training.ipynb
     │   └── model_evaluation.ipynb
     ├── scripts/
     │   ├── data_preparation.py
     │   ├── train_model.py
     │   └── evaluate_model.py
     ├── pipelines/
     │   ├── pipeline_definition.py
     │   └── pipeline_steps/
     │       ├── step1_preprocessing.py
     │       ├── step2_training.py
     │       └── step3_evaluation.py
     ├── mlflow/
     │   ├── conda.yaml
     │   ├── MLproject
     │   └── run_experiment.py
     ├── results/
     │   └── evaluation_metrics.json
     ├── report/
     │   └── final_report.* # extensión markdown o pdf
     ├── presentation/
     │   └── final_presentation.pptx
     └── demo/
        └── demo_video.md # link a su video en YouTube
     ```

4. **Subir Cambios**:
   - Incluir la información de los participantes en el README.md de su carpeta.
   - Subir todos los cambios al repositorio forkeado en GitHub.
   - Asegurarse de que todo el contenido esté correctamente organizado y accesible.
   - Verificar que todos los archivos requeridos están presentes y que la documentación está completa.
   - Hacer un [Pull Request](https://github.com/OscarSantosMu/azure-ml-projects/pulls) al repositorio original.

#### Entregables:
- Código fuente del proyecto, incluyendo scripts de preparación de datos, entrenamiento y evaluación del modelo.
- Informe escrito que detalle cada paso del proyecto, los resultados obtenidos y las conclusiones.
- Presentación final que resuma el proyecto, los resultados y las lecciones aprendidas.
- Video de una demostración práctica de la solución implementada con duración máxima de 1 minuto.

#### Recursos:
- Acceso a Azure Machine Learning y MLflow.
- Materiales y notas del bootcamp.
- Bibliografía recomendada sobre ciencia de datos y aprendizaje automático.
- Ejemplos y casos de estudio proporcionados durante el bootcamp.

#### ¿Y si tengo una pregunta?
Si tienes alguna pregunta, por favor abre un nuevo [issue](https://github.com/OscarSantosMu/azure-ml-projects/issues/new?assignees=OscarSantosMu&labels=&projects=&template=q-a.md&title=%5BQ%5D%3A+) en el repositorio con los detalles de tu consulta.