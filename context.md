# Amazon Recsys: contexto experimental

Este proyecto didáctico compara popularidad, ALS implícito, LightGCN y SASRec con reseñas de videojuegos de Amazon Reviews 2023. Prioriza implementaciones comprensibles y muestras manejables; no pretende ser un sistema de producción. Los pipelines se ejecutan con Kedro, Spark y PyTorch.

## Datos y protocolo

El procesamiento selecciona las columnas necesarias, elimina filas sin identificador, rating o fecha, descarta ratings fuera de 1–5 y conserva la reseña más reciente de cada par usuario–producto. Primero filtra usuarios y productos por actividad y después vuelve a eliminar usuarios con menos de tres interacciones. Indexa usuarios y productos y separa sus interacciones por tiempo: última para test, penúltima para validación y anteriores para entrenamiento. En empates se usa el índice de producto como criterio estable.

Este reparto es **retrospectivo por usuario**: el entrenamiento puede contener interacciones de otros usuarios posteriores al test de un usuario. No simula un corte global de tiempo ni permite afirmar que un producto estuviera disponible en el momento de cada predicción.

Se considera positiva una valoración de al menos 4. Todos los modelos se entrenan con positivos y comparten el catálogo de productos positivos de entrenamiento. Una interacción de validación o test es evaluable si es positiva, el usuario tiene al menos un positivo de entrenamiento y el producto pertenece al catálogo de entrenamiento. `experiment_stats` guarda el tamaño del catálogo y el número de objetivos elegibles, negativos, de usuarios sin historial positivo y de productos desconocidos; las categorías de exclusión son consecutivas y excluyentes.

Para recomendar en test se excluyen todos los productos del usuario en entrenamiento y validación, incluso ratings negativos. En validación se excluyen los vistos en entrenamiento. Cada métrica de ranking compara exactamente la misma población elegible y mide recall medio por usuario. El dashboard y el notebook leen `eligible_test` y calculan métricas sobre una muestra reproducible de esa población. La cobertura usa como denominador el catálogo positivo de entrenamiento.

## Modelos

- **Popularidad:** ordena productos por número de interacciones positivas de entrenamiento, luego rating medio y finalmente ID. El mismo orden global se filtra por lo visto por cada usuario antes de tomar el top K.
- **ALS implícito:** Spark factoriza la matriz de interacciones positivas binarias. Sus scores representan afinidad, no ratings de 1 a 5; por ello no se calcula RMSE de ratings. El número de candidatos cubre el catálogo entrenado antes de excluir vistos.
- **LightGCN:** usa un grafo bipartito normalizado y promedia embeddings iniciales y de cada capa de propagación. BPR compara un producto positivo con uno no positivo del usuario; la regularización se aplica a embeddings iniciales. Cada lote recalcula la propagación tras actualizar los pesos. La selección del checkpoint usa Recall@20 de validación.
- **SASRec:** aprende a predecir el siguiente positivo mediante un Transformer causal y negativos muestreados entre productos positivos de entrenamiento no asociados al usuario. Cada prefijo del historial genera un ejemplo; cada contexto se limita a los últimos 50 productos. La selección del checkpoint usa Recall@20 de validación. Para construir el historial de test se incorporan positivos de validación conocidos en entrenamiento; los productos desconocidos se omiten del contexto y nunca se convierten en padding silenciosamente.

SASRec utiliza `TransformerEncoderLayer` de PyTorch con GELU y entrena con prefijos individuales; es una variante didáctica y no una reproducción exacta de todos los detalles del artículo original. ALS se ejecuta con los hiperparámetros configurados, sin búsqueda automática de hiperparámetros.

Ambos modelos neuronales requieren al menos una interacción de validación elegible y un negativo disponible para cada usuario entrenado. Los checkpoints guardan el mejor estado según Recall@20, no necesariamente el último. Los empates de validación conservan el primer mejor estado.

## Uso y artefactos

Consulta el README para instalar dependencias y ejecutar los pipelines. `all_recommenders` ejecuta procesamiento, protocolo y los tres métodos. También existen pipelines individuales, que esperan los splits en el catálogo y vuelven a calcular el protocolo. El dashboard y el notebook necesitan `eligible_test`, `experiment_stats` y las tablas de recomendaciones generados por la versión actual: vuelve a ejecutar los pipelines tras actualizar el código, ya que los artefactos anteriores se produjeron con otro protocolo y no deben compararse con los nuevos.
